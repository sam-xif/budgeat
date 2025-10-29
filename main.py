import json
from agent import ResearchAgent
from dotenv import load_dotenv

def main():
    load_dotenv()
    print("Starting the research agent...")
    
    with open('sites.json', 'r') as f:
        sites_config = json.load(f)
    
    # Create the agent
    agent = ResearchAgent()
    
    # Run on the first configured site with a sample query
    site = sites_config['sites'][0]
    query = "laptop"
    
    print(f"\nSearching for '{query}' on {site['name']}...")
    print(f"URL: {site['url']}")
    print(f"Search selector: {site['search_bar_selector']}\n")
    
    result = agent.run(
        url=site['url'],
        search_selector=site['search_bar_selector'],
        product_query=query
    )
    
    print("\n" + "="*50)
    print("RESULT:")
    print("="*50)
    print(result)
    
    agent.shutdown()
    print("\nAgent finished.")

if __name__ == "__main__":
    main()
