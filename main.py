import os
import json
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import requests

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

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    try:
        return _ai_sentiment(request.comment)
    except Exception as e:
        fallback = _fallback_sentiment(request.comment)
        return SentimentResponse(sentiment=fallback.sentiment, rating=fallback.rating)

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