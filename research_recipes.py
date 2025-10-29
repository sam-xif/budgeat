#!/usr/bin/env python3
"""
Automated recipe price research script.
Usage: python research_recipes.py
"""

from agent import research_recipes
import json

# Example recipes
recipes = [
    {
        "name": "Breakfast Bowl",
        "ingredients": ["milk", "eggs", "bread"]
    },
    {
        "name": "Pasta Night",
        "ingredients": ["pasta", "tomato sauce", "ground beef", "cheese"]
    },
    {
        "name": "Sandwich Lunch",
        "ingredients": ["bread", "turkey", "cheese", "lettuce"]
    }
]

if __name__ == "__main__":
    print("Starting automated recipe price research...")
    print(f"Researching {len(recipes)} recipes...")
    
    results = research_recipes(recipes)
    
    # Print formatted results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for recipe in results:
        print(f"\n{recipe['name']} - Status: {recipe['status']}")
        print(f"Total: {recipe['total_price']}")
        print("Ingredients:")
        for ing in recipe['ingredients']:
            print(f"  • {ing['name']:20s} {ing['price']:10s} ({ing['site']})")
    
    # Save to JSON file
    with open('recipe_prices.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to recipe_prices.json")
