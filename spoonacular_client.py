"""
Spoonacular API client.

This module provides a high-level `SpoonacularClient` that authenticates using an API
key loaded from the environment (via `.env`) and exposes a few commonly used
endpoints with clear, well-documented parameters.

Environment variables:
  - SPOONACULAR_API_KEY: Your Spoonacular API key. Load from a `.env` file in the
    project root (or set in your shell environment).

Usage example:
  from spoonacular_client import SpoonacularClient

  client = SpoonacularClient()  # loads key from .env / environment
  results = client.search_recipes(query="chicken pasta", number=5, add_recipe_information=True)
  print(results)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests
from dotenv import load_dotenv
import os


DEFAULT_BASE_URL = "https://api.spoonacular.com"


@dataclass
class SpoonacularClientConfig:
    """Configuration for `SpoonacularClient`.

    Attributes:
        api_key: Spoonacular API key. If None, loaded from environment variable
            `SPOONACULAR_API_KEY` after calling `load_dotenv()`.
        base_url: Base URL for the Spoonacular API.
        timeout_seconds: Per-request timeout in seconds.
        max_retries: Number of retries for transient errors (HTTP 429/5xx).
        retry_backoff_seconds: Initial backoff delay (exponential) between retries.
    """

    api_key: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0


class SpoonacularClient:
    """Typed client wrapper for the Spoonacular REST API.

    The client adds the `apiKey` query parameter automatically for each request and
    performs basic retry handling for 429/5xx responses.

    Initialize with an explicit API key, or rely on `.env`/environment:
        client = SpoonacularClient()  # loads SPOONACULAR_API_KEY

    Notes on authentication:
    - Spoonacular accepts `apiKey` as a query parameter. Some endpoints may also accept
      `x-api-key` header, but `apiKey` in the query is most universal; this client uses
      the query parameter.
    """

    def __init__(self, config: Optional[SpoonacularClientConfig] = None):
        if config is None:
            config = SpoonacularClientConfig()

        # Load env first if api_key not provided explicitly
        if config.api_key is None:
            load_dotenv()
            config.api_key = os.getenv("SPOONACULAR_API_KEY")

        if not config.api_key:
            raise ValueError(
                "SPOONACULAR_API_KEY not found. Set it in your environment or .env file."
            )

        self._config = config
        self._session = requests.Session()

    # -----------------------------
    # Internal request helper
    # -----------------------------
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self._config.base_url.rstrip('/')}/{path.lstrip('/')}"

        # Inject apiKey query param
        merged_params: Dict[str, Any] = {"apiKey": self._config.api_key}
        if params:
            merged_params.update({k: v for k, v in params.items() if v is not None})

        backoff = self._config.retry_backoff_seconds
        for attempt in range(self._config.max_retries + 1):
            response = self._session.request(
                method=method.upper(),
                url=url,
                params=merged_params,
                json=json,
                timeout=self._config.timeout_seconds,
            )

            if response.status_code in (429, 500, 502, 503, 504) and attempt < self._config.max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue

            response.raise_for_status()
            # Try JSON, fall back to text
            try:
                return response.json()
            except ValueError:
                return response.text

    # --------------------------------------------------
    # Public API methods (commonly used endpoints)
    # --------------------------------------------------
    def search_recipes(
        self,
        *,
        query: Optional[str] = None,
        cuisine: Optional[str] = None,
        diet: Optional[str] = None,
        intolerances: Optional[Iterable[str]] = None,
        include_ingredients: Optional[Iterable[str]] = None,
        exclude_ingredients: Optional[Iterable[str]] = None,
        type: Optional[str] = None,
        max_ready_time: Optional[int] = None,
        min_calories: Optional[int] = None,
        max_calories: Optional[int] = None,
        number: int = 10,
        offset: int = 0,
        add_recipe_information: bool = False,
        sort: Optional[str] = None,
        sort_direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search recipes using the `complexSearch` endpoint.

        Parameters:
          - query: Free-text recipe search query (e.g., "chicken pasta").
          - cuisine: Comma-separated cuisines (e.g., "italian,american").
          - diet: One of Spoonacular diets (e.g., "vegetarian", "vegan", "keto").
          - intolerances: Iterable of intolerances (e.g., ["gluten", "dairy"]).
          - include_ingredients: Ingredients that must be included.
          - exclude_ingredients: Ingredients that must not be included.
          - type: Dish type (e.g., "main course", "dessert").
          - max_ready_time: Maximum ready time in minutes.
          - min_calories/max_calories: Calorie bounds for results.
          - number: Page size (default 10, max 100 depending on plan).
          - offset: Pagination offset.
          - add_recipe_information: If True, include additional fields (costs more quota).
          - sort: Sort field (e.g., "popularity", "healthiness", "time", "price").
          - sort_direction: "asc" or "desc".

        Returns: JSON dict from Spoonacular (contains `results` and `totalResults`).
        """

        def _join(values: Optional[Iterable[str]]) -> Optional[str]:
            if values is None:
                return None
            return ",".join([v for v in values if v]) or None

        params: Dict[str, Any] = {
            "query": query,
            "cuisine": cuisine,
            "diet": diet,
            "intolerances": _join(intolerances),
            "includeIngredients": _join(include_ingredients),
            "excludeIngredients": _join(exclude_ingredients),
            "type": type,
            "maxReadyTime": max_ready_time,
            "minCalories": min_calories,
            "maxCalories": max_calories,
            "number": number,
            "offset": offset,
            "addRecipeInformation": add_recipe_information,
            "sort": sort,
            "sortDirection": sort_direction,
        }
        return self._request("GET", "/recipes/complexSearch", params=params)

    def get_recipe_information(
        self,
        recipe_id: int,
        *,
        include_nutrition: bool = False,
    ) -> Dict[str, Any]:
        """Get detailed information about a recipe.

        Parameters:
          - recipe_id: The Spoonacular recipe ID.
          - include_nutrition: If True, includes full nutrition in the response.

        Returns: JSON dict for the recipe.
        """
        params = {"includeNutrition": include_nutrition}
        return self._request("GET", f"/recipes/{recipe_id}/information", params=params)

    def get_recipe_nutrition_widget(
        self,
        recipe_id: int,
        *,
        format: str = "json",
    ) -> Any:
        """Get nutrition widget data for a recipe.

        Parameters:
          - recipe_id: The Spoonacular recipe ID.
          - format: "json" or "html" (defaults to "json").

        Returns: JSON dict or HTML string depending on format.
        """
        path = f"/recipes/{recipe_id}/nutritionWidget.{format.lower()}"
        return self._request("GET", path)

    def parse_ingredients(
        self,
        ingredients: Iterable[str],
        *,
        servings: int = 1,
        include_nutrition: bool = False,
        language: str = "en",
    ) -> List[Dict[str, Any]]:
        """Parse raw ingredient lines into structured data.

        Parameters:
          - ingredients: Iterable of raw ingredient strings (e.g., "2 cups flour").
          - servings: Number of servings the recipe makes (affects nutrition).
          - include_nutrition: If True, include nutrition per ingredient.
          - language: Language code (e.g., "en").

        Returns: List of parsed ingredient dicts.
        """
        lines: List[str] = [s for s in ingredients if s]
        body = {
            "ingredientList": "\n".join(lines),
            "servings": servings,
            "includeNutrition": include_nutrition,
            "language": language,
        }
        return self._request("POST", "/recipes/parseIngredients", json=body)

    def get_random_recipes(
        self,
        *,
        number: int = 1,
        tags: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Get random recipes.

        Parameters:
          - number: Number of recipes to return.
          - tags: Optional tags filter (e.g., ["vegetarian", "dessert"]).

        Returns: JSON dict with `recipes`.
        """
        def _join(values: Optional[Iterable[str]]) -> Optional[str]:
            if values is None:
                return None
            return ",".join([v for v in values if v]) or None

        params = {"number": number, "tags": _join(tags)}
        return self._request("GET", "/recipes/random", params=params)

    def autocomplete_ingredient_search(
        self,
        *,
        query: str,
        number: int = 10,
        meta_information: bool = False,
    ) -> List[Dict[str, Any]]:
        """Autocomplete ingredient names.

        Parameters:
          - query: The partial ingredient name to search for.
          - number: Max number of results.
          - meta_information: If True, include extra metadata for each result.

        Returns: List of ingredient suggestion dicts.
        """
        params = {
            "query": query,
            "number": number,
            "metaInformation": meta_information,
        }
        return self._request("GET", "/food/ingredients/autocomplete", params=params)


def get_spoonacular_client(config: Optional[SpoonacularClientConfig] = None) -> SpoonacularClient:
    """Convenience factory that returns a `SpoonacularClient`.

    Parameters:
      - config: Optional `SpoonacularClientConfig`. If omitted, sensible defaults are used
        and the API key is loaded from the environment (`SPOONACULAR_API_KEY`).
    """
    return SpoonacularClient(config=config)


def get_ingredients_for_query(query: str, *, number: int = 10) -> Dict[str, List[str]]:
    """Return a mapping from recipe title to list of ingredient strings for a query.

    This helper performs two steps:
      1) Calls `complexSearch` to get up to `number` recipes for the given query.
      2) For each recipe, fetches `information` and extracts `extendedIngredients`.

    Parameters:
      - query: Free-text search string (e.g., "pasta", "chicken curry").
      - number: Maximum number of recipes to retrieve (default 10).

    Returns:
      Dict where keys are recipe titles and values are lists of ingredient strings
      in their original, human-readable form (e.g., "2 cups flour").
    """
    client = SpoonacularClient()
    search = client.search_recipes(query=query, number=number, add_recipe_information=False)
    results = search.get("results") or []

    mapping: Dict[str, List[str]] = {}
    for item in results:
        recipe_id = item.get("id")
        if not isinstance(recipe_id, int):
            continue

        info = client.get_recipe_information(recipe_id)
        title = info.get("title") or f"Recipe {recipe_id}"
        ingredients = []
        for ing in info.get("extendedIngredients", []) or []:
            if isinstance(ing, dict):
                original = ing.get("name") or ing.get("originalString")
                if original:
                    ingredients.append(original)
        mapping[title] = ingredients

    return mapping


