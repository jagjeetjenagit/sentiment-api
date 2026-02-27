const { chromium } = require('playwright');

const urls = [
    'https://sanand0.github.io/tdsdata/js_table/?seed=89',
    'https://sanand0.github.io/tdsdata/js_table/?seed=90',
    'https://sanand0.github.io/tdsdata/js_table/?seed=91',
    'https://sanand0.github.io/tdsdata/js_table/?seed=92',
    'https://sanand0.github.io/tdsdata/js_table/?seed=93',
    'https://sanand0.github.io/tdsdata/js_table/?seed=94',
    'https://sanand0.github.io/tdsdata/js_table/?seed=95',
    'https://sanand0.github.io/tdsdata/js_table/?seed=96',
    'https://sanand0.github.io/tdsdata/js_table/?seed=97',
    'https://sanand0.github.io/tdsdata/js_table/?seed=98'
];

async function scrapeAndSum() {
    const browser = await chromium.launch();
    const page = await browser.newPage();
    
    let grandTotal = 0;
    
    console.log('Starting to scrape tables from all URLs...\n');
    
    for (const url of urls) {
        console.log(`Scraping: ${url}`);
        
        await page.goto(url, { waitUntil: 'networkidle' });
        
        // Wait a bit for any dynamic content to load
        await page.waitForTimeout(2000);
        
        // Extract all numbers from all table cells
        const pageSum = await page.evaluate(() => {
            let sum = 0;
            const cells = document.querySelectorAll('table td, table th');
            
            cells.forEach(cell => {
                const text = cell.textContent.trim();
                const numbers = text.match(/\d+/g);
                
                if (numbers) {
                    numbers.forEach(numStr => {
                        const num = parseInt(numStr, 10);
                        if (!isNaN(num)) {
                            sum += num;
                        }
                    });
                }
            });
            
            return sum;
        });
        
        console.log(`  Sum for this page: ${pageSum}`);
        grandTotal += pageSum;
    }
    
    await browser.close();
    
    console.log('\n' + '='.repeat(80));
    console.log('GRAND TOTAL OF ALL NUMBERS FROM ALL TABLES:');
    console.log(grandTotal);
    console.log('='.repeat(80));
    
    return grandTotal;
}

scrapeAndSum()
    .then(total => {
        console.log(`\n✓ Successfully scraped and summed all tables. Total: ${total}`);
        process.exit(0);
    })
    .catch(error => {
        console.error('✗ Error:', error);
        process.exit(1);
    });
