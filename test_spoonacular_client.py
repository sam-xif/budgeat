"""
Quick runtime test for `SpoonacularClient`.

Run:
  python test_spoonacular_client.py

Requirements:
  - `.env` file with SPOONACULAR_API_KEY set, or env var in shell.
"""

from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

from spoonacular_client import SpoonacularClient, get_ingredients_for_query


def main() -> int:
    load_dotenv()
    api_key = os.getenv("SPOONACULAR_API_KEY")
    if not api_key:
        print("ERROR: SPOONACULAR_API_KEY not found in environment or .env")
        return 1

    client = SpoonacularClient()

    # 1) Basic search
    print("Searching for 'pasta' recipes (1 result)...")
    search = client.search_recipes(
        query="pasta", 
        number=10, 
        add_recipe_information=True)
    if not isinstance(search, dict) or "results" not in search or not search["results"]:
        print("ERROR: Unexpected search response:", search)
        return 1

    first = search["results"][0]
    recipe_id = first.get("id")
    title = first.get("title")
    print(f"Found recipe: id={recipe_id}, title={title}")

    if not isinstance(recipe_id, int):
        print("ERROR: Missing/invalid recipe id in search result")
        return 1

    # 2) Fetch recipe information
    print(f"Fetching information for recipe {recipe_id}...")
    info = client.get_recipe_information(recipe_id)
    if not isinstance(info, dict) or info.get("id") != recipe_id:
        print("ERROR: Unexpected recipe info response:", info)
        return 1

    print("Success! Client calls are working:")
    print(f"- Recipe title: {info.get('title')}")
    print(f"- Ready in minutes: {info.get('readyInMinutes')}")
    print(f"- Servings: {info.get('servings')}")
    # print(f"- Ingredients: {info.get('extendedIngredients')}")
    print(f"- Ingredients: {[ingredient.get('name') for ingredient in info.get('extendedIngredients')]}")
    print(f"- Nutrition: {info.get('nutrition')}")

    # 3) Helper usage: mapping from recipe title -> ingredient list (original strings)
    print("\nFetching ingredients mapping for query 'pasta' (up to 10 recipes)...")
    mapping = get_ingredients_for_query("pasta", number=10)
    if not isinstance(mapping, dict) or not mapping:
        print("ERROR: Unexpected mapping result:", mapping)
        return 1

    # Show a sample
    sample_title = next(iter(mapping))
    print(f"Sample recipe: {sample_title}")
    for ing in mapping[sample_title][:5]:
        print(f"  - {ing}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


