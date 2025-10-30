"""
USDA FoodData Central API integration for calorie data.
"""

import os
import requests


def get_usda_calories(ingredient_name: str) -> dict:
    """
    Get calorie information from USDA FoodData Central API.
    
    Args:
        ingredient_name: Name of the ingredient to look up
    
    Returns:
        {"calories": 150, "serving_size": "100g", "found": True}
    """
    try:
        # USDA FoodData Central API - free, no key required for basic search
        search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        params = {
            "query": ingredient_name,
            "pageSize": 1,
            "api_key": "DEMO_KEY"  # Free demo key, or set USDA_API_KEY env var
        }
        
        # Check if user has their own API key
        api_key = os.getenv("USDA_API_KEY")
        if api_key:
            params["api_key"] = api_key
        
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("foods") and len(data["foods"]) > 0:
            food = data["foods"][0]
            nutrients = food.get("foodNutrients", [])
            
            # Find calories (Energy)
            for nutrient in nutrients:
                if nutrient.get("nutrientName") == "Energy" and nutrient.get("unitName") == "KCAL":
                    return {
                        "calories": int(nutrient.get("value", 0)),
                        "serving_size": "100g",
                        "found": True
                    }
        
        return {"calories": None, "serving_size": None, "found": False}
    
    except Exception as e:
        print(f"Error fetching USDA data for {ingredient_name}: {e}")
        return {"calories": None, "serving_size": None, "found": False}
