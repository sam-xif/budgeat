# BudgEat 🥗

AI-powered grocery price research with vision capabilities. Uses NVIDIA's Nemotron VLM to search multiple stores and find the best prices.

## Features

- 🤖 **Agentic AI** - Uses LangGraph with NVIDIA Nemotron for intelligent decision-making
- 👁️ **Vision AI** - Can literally read page screenshots when JavaScript renders content
- 🏪 **Multi-store** - Searches Target, Amazon, Walmart, Kroger automatically
- ⚡ **Fast** - Optimized with headless browsing and smart waits
- 📊 **Batch Processing** - Research entire recipes automatically

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Set your NVIDIA API key in .env
echo "NVIDIA_API_KEY=your_key_here" > .env
```

### 2. Run the Web UI

```bash
streamlit run streamlit_app.py
```

Then search for products on various stores with real-time AI progress tracking.

### 3. Batch Recipe Research (Automated)

```bash
python research_recipes.py
```

Or use programmatically:

```python
from agent import research_recipes

recipes = [
    {
        "name": "Breakfast Bowl",
        "ingredients": ["milk", "eggs", "bread"]
    },
    {
        "name": "Pasta Night",
        "ingredients": ["pasta", "tomato sauce", "ground beef"]
    }
]

results = research_recipes(recipes)
# Returns structured price data for each recipe
```

## How It Works

1. **Navigate** - Goes directly to search URLs (e.g., `target.com/s?searchTerm=milk`)
2. **Wait for JS** - Waits for dynamic content to load
3. **Try HTML parsing** - Attempts to extract text from page
4. **Fall back to Vision** - If HTML is incomplete, sends screenshot to NVIDIA's VLM
5. **Extract prices** - AI reads and understands the content to find relevant prices

## Architecture

- **LangGraph** - Agentic reasoning framework
- **NVIDIA Nemotron** - Vision-language model for understanding content
- **Playwright** - Browser automation with stealth mode
- **BeautifulSoup** - HTML parsing
- **Streamlit** - Interactive web UI
