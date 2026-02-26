import os
import json
import re
import sys
import time
import glob
import shutil
import tempfile
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Literal, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv
import requests
import yt_dlp

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class CommentRequest(BaseModel):
    comment: str

# Response model
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int


class CodeExecutionRequest(BaseModel):
    code: str


class CodeInterpreterResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


class AskRequest(BaseModel):
    video_url: str
    topic: str


class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


class TimestampResult(BaseModel):
    timestamp: str


@app.get("/")
async def root():
    return {"status": "ok", "service": "sentiment-api"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


def _fallback_sentiment(comment: str) -> SentimentResponse:
    text = comment.lower()

    positive_words = [
        "amazing", "great", "excellent", "good", "love", "awesome", "fantastic", "happy", "best"
    ]
    negative_words = [
        "bad", "terrible", "awful", "worst", "hate", "poor", "disappointing", "horrible", "sad"
    ]

    positive_score = sum(1 for word in positive_words if word in text)
    negative_score = sum(1 for word in negative_words if word in text)

    if positive_score > negative_score:
        sentiment = "positive"
        rating = 5 if positive_score >= 2 else 4
    elif negative_score > positive_score:
        sentiment = "negative"
        rating = 1 if negative_score >= 2 else 2
    else:
        sentiment = "neutral"
        rating = 3

    return SentimentResponse(sentiment=sentiment, rating=rating)


def _ai_sentiment(comment: str) -> SentimentResponse:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a sentiment classifier. "
                    "Return only valid JSON with keys: sentiment and rating."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Classify this comment sentiment as positive, negative, or neutral. "
                    "Return rating 1-5 where 5 is highly positive and 1 is highly negative.\n\n"
                    f"Comment: {comment}"
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral"]
                        },
                        "rating": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5
                        }
                    },
                    "required": ["sentiment", "rating"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
    )

    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Model returned empty content")

    parsed = json.loads(content)
    return SentimentResponse(**parsed)


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    buffer = StringIO()

    try:
        sys.stdout = buffer
        sys.stderr = buffer
        with redirect_stdout(buffer), redirect_stderr(buffer):
            exec(code, {})
        output = buffer.getvalue()
        return {"success": True, "output": output}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _extract_traceback_lines(traceback_text: str) -> List[int]:
    matches = re.findall(r'File "<string>", line (\d+)', traceback_text)
    lines = sorted({int(value) for value in matches})
    return lines


def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _extract_traceback_lines(traceback_text)

    client = genai.Client(api_key=api_key)

    prompt = f"""
Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{traceback_text}

Return the line number(s) where the error is located.
"""

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER),
                    )
                },
                required=["error_lines"],
            ),
        ),
    )

    result = ErrorAnalysis.model_validate_json(response.text)
    if not result.error_lines:
        return _extract_traceback_lines(traceback_text)
    return result.error_lines


def _download_audio_only(video_url: str) -> tuple[str, str]:
    temp_dir = tempfile.mkdtemp(prefix="yt_audio_")
    output_template = os.path.join(temp_dir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        audio_path = ydl.prepare_filename(info)

    if not os.path.exists(audio_path):
        candidates = sorted(glob.glob(os.path.join(temp_dir, "audio.*")))
        if not candidates:
            raise RuntimeError("Audio download failed")
        audio_path = candidates[0]

    return temp_dir, audio_path


def _wait_for_file_active(gemini_client: genai.Client, file_name: str, timeout_seconds: int = 300) -> object:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        file_obj = gemini_client.files.get(name=file_name)
        state_obj = getattr(file_obj, "state", None)
        state_name = getattr(state_obj, "name", str(state_obj))

        if state_name == "ACTIVE":
            return file_obj

        if state_name in {"FAILED", "ERROR", "CANCELLED"}:
            raise RuntimeError(f"Gemini file processing failed with state: {state_name}")

        time.sleep(2)

    raise RuntimeError("Timed out waiting for Gemini file to become ACTIVE")


def _find_topic_timestamp(video_url: str, topic: str) -> str:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    temp_dir = None
    try:
        temp_dir, audio_path = _download_audio_only(video_url)

        gemini_client = genai.Client(api_key=gemini_api_key)
        uploaded = gemini_client.files.upload(file=audio_path)
        active_file = _wait_for_file_active(gemini_client, uploaded.name)

        prompt = f"""
You are analyzing spoken audio from a YouTube video.
Find the FIRST moment when this topic or phrase is spoken: {topic}

Return exactly one timestamp in HH:MM:SS format.
Do not return explanations.
"""

        response = gemini_client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            contents=[active_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "timestamp": types.Schema(
                            type=types.Type.STRING,
                            pattern=r"^\\d{2}:\\d{2}:\\d{2}$",
                        )
                    },
                    required=["timestamp"],
                ),
            ),
        )

        result = TimestampResult.model_validate_json(response.text)
        if not re.match(r"^\d{2}:\d{2}:\d{2}$", result.timestamp):
            raise RuntimeError("Invalid timestamp format returned by model")

        return result.timestamp
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    try:
        return _ai_sentiment(request.comment)
    except Exception as e:
        fallback = _fallback_sentiment(request.comment)
        return SentimentResponse(sentiment=fallback.sentiment, rating=fallback.rating)


@app.post("/code-interpreter", response_model=CodeInterpreterResponse)
async def code_interpreter(request: CodeExecutionRequest):
    execution_result = execute_python_code(request.code)

    if execution_result["success"]:
        return CodeInterpreterResponse(error=[], result=execution_result["output"])

    error_lines = analyze_error_with_ai(request.code, execution_result["output"])
    return CodeInterpreterResponse(error=error_lines, result=execution_result["output"])


@app.post("/ask", response_model=AskResponse)
async def ask_video(request: AskRequest):
    timestamp = _find_topic_timestamp(request.video_url, request.topic)
    return AskResponse(
        timestamp=timestamp,
        video_url=request.video_url,
        topic=request.topic,
    )

def analyze_sentiment(comment):
    url = "https://sentiment-api-production-f100.up.railway.app/comment"
    payload = {"comment": comment}  # Ensure the payload matches the expected schema

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

if __name__ == "__main__":
    comment = "This product is amazing!"
    result = analyze_sentiment(comment)
    print("Response:", result)