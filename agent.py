import os
import requests
import re
import base64
from typing import Optional, List, Any, Annotated
from playwright.sync_api import sync_playwright, Page, Browser, Playwright
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent

# Load environment variables from .env file
load_dotenv()


# Store page HTML and state in thread-safe way
class SharedBrowserState:
    """Stores browser state that can be shared across tool calls."""
    def __init__(self):
        self.current_html: Optional[str] = None
        self.current_url: Optional[str] = None
        self.last_action: Optional[str] = None
        self.last_screenshot: Optional[str] = None

SHARED_STATE = SharedBrowserState()


def get_browser_context():
    """Create a new browser context for each tool operation with stealth settings."""
    playwright = sync_playwright().start()
    
    # Launch with stealth settings
    browser = playwright.chromium.launch(
        headless=False,  # Non-headless is less detectable
        args=[
            '--disable-blink-features=AutomationControlled',
            '--disable-dev-shm-usage',
            '--no-sandbox',
        ]
    )
    
    # Create context with realistic settings
    context = browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        locale='en-US',
        timezone_id='America/New_York'
    )
    
    page = context.new_page()
    
    # Add extra headers to look more human
    page.set_extra_http_headers({
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://www.google.com/'
    })
    
    # If we have a previous state, restore it
    if SHARED_STATE.current_html and SHARED_STATE.current_url:
        page.goto('about:blank')
        page.set_content(SHARED_STATE.current_html)
    
    return playwright, browser, page


def save_browser_state(page: Page):
    """Save current browser state."""
    try:
        SHARED_STATE.current_html = page.content()
        SHARED_STATE.current_url = page.url
        
        # Also take a screenshot for vision analysis
        screenshot_path = "/tmp/budgeat_screenshot.png"
        page.screenshot(path=screenshot_path, full_page=False)  # Just viewport
        SHARED_STATE.last_screenshot = screenshot_path
    except Exception as e:
        print(f"Could not save browser state: {e}")


def navigate_to_url(url: str) -> str:
    """Navigate to a specific URL in the browser."""
    playwright, browser, page = get_browser_context()
    try:
        print(f"Navigating to {url}...")
        page.goto(url, wait_until='domcontentloaded', timeout=90000)
        
        # Wait for JavaScript to execute and load dynamic content
        page.wait_for_load_state('networkidle', timeout=30000)
        
        # Extra wait for lazy-loaded content
        page.wait_for_timeout(5000)
        
        save_browser_state(page)
        return f"Successfully navigated to {url}. Page loaded with dynamic content."
    except Exception as e:
        return f"Error navigating to {url}: {str(e)}"
    finally:
        browser.close()
        playwright.stop()


def find_and_fill_input(selector: str, text: str) -> str:
    """Find an input field by CSS selector and fill it with text."""
    playwright, browser, page = get_browser_context()
    try:
        print(f"Filling input '{selector}' with '{text}'...")
        page.locator(selector).fill(text)
        save_browser_state(page)
        return f"Successfully filled input '{selector}' with '{text}'"
    except Exception as e:
        return f"Error filling input '{selector}': {str(e)}"
    finally:
        browser.close()
        playwright.stop()


def click_element(selector: str) -> str:
    """Click an element by CSS selector (button, link, etc)."""
    playwright, browser, page = get_browser_context()
    try:
        print(f"Clicking element '{selector}'...")
        page.locator(selector).click()
        page.wait_for_timeout(2000)
        save_browser_state(page)
        return f"Successfully clicked element '{selector}'"
    except Exception as e:
        return f"Error clicking element '{selector}': {str(e)}"
    finally:
        browser.close()
        playwright.stop()


def press_enter(selector: str) -> str:
    """Press Enter key on an element (useful for search forms)."""
    playwright, browser, page = get_browser_context()
    try:
        print(f"Pressing Enter on '{selector}'...")
        page.press(selector, 'Enter')
        page.wait_for_timeout(5000)
        save_browser_state(page)
        return f"Successfully pressed Enter on '{selector}'. Waiting for results to load..."
    except Exception as e:
        return f"Error pressing Enter on '{selector}': {str(e)}"
    finally:
        browser.close()
        playwright.stop()


def read_page_content() -> str:
    """Get the current page's HTML content."""
    try:
        if SHARED_STATE.current_html:
            return f"Page content retrieved. Length: {len(SHARED_STATE.current_html)} characters. Use extract_prices_and_products to parse it."
        return "No page content available. Navigate to a URL first."
    except Exception as e:
        return f"Error reading page content: {str(e)}"


def extract_prices_and_products() -> str:
    """Get the page content for the AI to parse. Returns cleaned HTML text."""
    try:
        if not SHARED_STATE.current_html:
            return "No page loaded. Navigate to a URL first."
        
        html_content = SHARED_STATE.current_html
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = '\n'.join(lines)
        
        # Return first 15000 characters to avoid token limits
        if len(cleaned_text) > 15000:
            return cleaned_text[:15000] + "\n\n[... content truncated ...]"
        
        return cleaned_text
    
    except Exception as e:
        return f"Error reading page content: {str(e)}"


def analyze_screenshot_with_vision(query: str = "List all products visible with their prices") -> str:
    """Use NVIDIA's vision model to analyze the screenshot and extract product information."""
    try:
        if not SHARED_STATE.last_screenshot or not os.path.exists(SHARED_STATE.last_screenshot):
            return "No screenshot available. Navigate to a page first."
        
        # Encode screenshot as base64
        with open(SHARED_STATE.last_screenshot, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        
        # Call NVIDIA API with vision
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            return "NVIDIA_API_KEY not set"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        payload = {
            "model": "nvidia/nemotron-nano-12b-v2-vl",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2048,
            "stream": False
        }
        
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    except Exception as e:
        return f"Error analyzing screenshot: {str(e)}"

class ResearchAgent:
    """An agentic researcher that uses specialized tools to find product prices."""
    
    def __init__(self, model="nvidia/nemotron-nano-12b-v2-vl"):
        """
        Initialize the research agent.
        
        Args:
            model: NVIDIA model to use. Default is nvidia/nemotron-nano-12b-v2-vl
                   Check https://build.nvidia.com/models for available models.
        """
        # Get NVIDIA API key
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY environment variable not set.")
        
        # Initialize ChatOpenAI with NVIDIA endpoint
        # NVIDIA's API is OpenAI-compatible, so we can use ChatOpenAI
        self.llm = ChatOpenAI(
            model=model,
            api_key=nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.7,
            max_tokens=2048
        )
        
        # Define tools as simple functions with descriptions
        # LangGraph will automatically wrap them
        from langchain_core.tools import tool as tool_decorator
        
        @tool_decorator
        def search_product_on_site(base_url: str, query: str) -> str:
            """
            Search for a product on an e-commerce site by constructing a direct search URL.
            Supports: Target, Amazon, Walmart, Kroger
            """
            # Build search URL based on the site
            encoded_query = query.replace(' ', '+')
            
            if 'target' in base_url.lower():
                search_url = f"{base_url}/s?searchTerm={encoded_query}"
            elif 'amazon' in base_url.lower():
                search_url = f"{base_url}/s?k={encoded_query}"
            elif 'walmart' in base_url.lower():
                search_url = f"{base_url}/search?q={encoded_query}"
            elif 'kroger' in base_url.lower():
                search_url = f"{base_url}/search?query={encoded_query}"
            else:
                # Generic approach
                search_url = f"{base_url}/search?q={encoded_query}"
            
            return navigate_to_url(search_url)
        
        @tool_decorator
        def get_page_content() -> str:
            """
            Get the text content of the current page for analysis.
            Returns cleaned page text that you should parse to find product names and prices.
            """
            return extract_prices_and_products()
        
        @tool_decorator
        def read_page_visually(query: str = "List all products with their names and prices that you can see on this page") -> str:
            """
            Use vision AI to read the current page screenshot and extract product information.
            This is useful for pages with dynamic JavaScript content.
            You can customize the query to ask specific questions about what's visible.
            """
            return analyze_screenshot_with_vision(query)
        
        self.tools = [search_product_on_site, get_page_content, read_page_visually]
        
        # Create the ReAct agent using LangGraph's prebuilt function
        self.agent_executor = create_react_agent(self.llm, self.tools)

    def run(self, url: str, search_selector: str, product_query: str):
        """
        Run the agent to find a product price.
        
        Args:
            url: The website URL to search (e.g., 'https://www.walmart.com')
            search_selector: Not used anymore (kept for compatibility)
            product_query: The product to search for (e.g., 'laptop')
        """
        question = (
            f'Find the price for "{product_query}" on {url}.\n\n'
            f'Instructions:\n'
            f'1. Use search_product_on_site with base_url="{url}" and query="{product_query}"\n'
            f'2. Try get_page_content first to retrieve the page text\n'
            f'3. If the page content seems incomplete or has no products, use read_page_visually instead\n'
            f'   (this uses vision AI to literally read the screenshot - great for JavaScript pages)\n'
            f'4. Identify products matching "{product_query}" and extract names and prices\n'
            f'5. Report the most relevant product with its price\n\n'
            f'Note: For sites like Target, use read_page_visually since they load products with JavaScript.'
        )

        try:
            # LangGraph agents use messages
            result = self.agent_executor.invoke({"messages": [HumanMessage(content=question)]})
            # Extract the final AI message
            messages = result["messages"]
            final_message = messages[-1]
            return final_message.content if hasattr(final_message, 'content') else str(final_message)
        except Exception as e:
            import traceback
            return f"Error during research: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

    def run_with_progress(self, url: str, search_selector: str, product_query: str, progress_callback=None):
        """
        Run the agent with progress updates.
        
        Args:
            url: The website URL to search
            search_selector: Not used anymore (kept for compatibility)
            product_query: The product to search for
            progress_callback: Optional callback function to receive progress updates
        """
        question = (
            f'Find the price for "{product_query}" on {url}.\n\n'
            f'Instructions:\n'
            f'1. Use search_product_on_site with base_url="{url}" and query="{product_query}"\n'
            f'2. Try get_page_content first to retrieve the page text\n'
            f'3. If the page content seems incomplete or has no products, use read_page_visually instead\n'
            f'   (this uses vision AI to literally read the screenshot - great for JavaScript pages)\n'
            f'4. Identify products matching "{product_query}" and extract names and prices\n'
            f'5. Report the most relevant product with its price\n\n'
            f'Note: For sites like Target, use read_page_visually since they load products with JavaScript.'
        )

        try:
            # Stream the agent's execution
            messages_so_far = []
            for chunk in self.agent_executor.stream({"messages": [HumanMessage(content=question)]}):
                if progress_callback and 'agent' in chunk:
                    # Extract messages from the agent step
                    agent_messages = chunk['agent'].get('messages', [])
                    for msg in agent_messages:
                        if hasattr(msg, 'content') and msg.content and msg not in messages_so_far:
                            # Show the agent's thoughts or tool calls
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    progress_callback(f"🔧 Using tool: **{tool_call['name']}**")
                            elif msg.content:
                                progress_callback(f"💭 Agent: {msg.content[:200]}...")
                            messages_so_far.append(msg)
                
                if progress_callback and 'tools' in chunk:
                    # Show tool execution results
                    tool_messages = chunk['tools'].get('messages', [])
                    for msg in tool_messages:
                        if hasattr(msg, 'content') and msg.content:
                            progress_callback(f"📤 Tool result: {msg.content[:150]}...")
            
            # Get the final result
            result = self.agent_executor.invoke({"messages": [HumanMessage(content=question)]})
            messages = result["messages"]
            final_message = messages[-1]
            return final_message.content if hasattr(final_message, 'content') else str(final_message)
            
        except Exception as e:
            import traceback
            return f"Error during research: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

    def shutdown(self):
        """Clean up browser resources."""
        # Clear shared state
        SHARED_STATE.current_html = None
        SHARED_STATE.current_url = None
        SHARED_STATE.last_action = None
        print("Cleared browser state.")
