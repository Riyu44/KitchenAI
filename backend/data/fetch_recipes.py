"""
fetch_recipes.py
Fetches recipes from Spoonacular API and stores them in Supabase.
Run once to populate the recipes table.
Usage: python fetch_recipes.py
"""

import os
import time
import requests
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SPOONACULAR_KEY = os.getenv("SPOONACULAR_API_KEY")
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

BASE_URL = "https://api.spoonacular.com"

# Cuisine and meal type combinations to fetch
CUISINES = [
    "Indian", "Chinese", "Italian", "Mexican",
    "Mediterranean", "Thai", "Japanese", "American"
]

MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_get(url, params, retries=3):
    """GET with retry on failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 402:
                print("  Daily API limit reached. Run again tomorrow.")
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return None


def parse_ingredients(extended_ingredients):
    """Convert Spoonacular ingredient format to our schema."""
    ingredients = []
    for ing in extended_ingredients:
        ingredients.append({
            "name":     ing.get("nameClean") or ing.get("name", ""),
            "quantity": ing.get("amount", 0),
            "unit":     ing.get("unit", ""),
            "original": ing.get("original", ""),
        })
    return ingredients


def fetch_recipe_details(recipe_id):
    """Fetch full recipe details including ingredients and instructions."""
    data = safe_get(
        f"{BASE_URL}/recipes/{recipe_id}/information",
        params={
            "apiKey": SPOONACULAR_KEY,
            "includeNutrition": False,
        }
    )
    return data


def fetch_recipes_by_cuisine(cuisine, meal_type, num=10):
    """Search for recipes by cuisine and meal type."""
    data = safe_get(
        f"{BASE_URL}/recipes/complexSearch",
        params={
            "apiKey":    SPOONACULAR_KEY,
            "cuisine":   cuisine,
            "type":      meal_type,
            "number":    num,
            "addRecipeInformation": True,
        }
    )
    if not data:
        return []
    return data.get("results", [])


def store_recipe(recipe_data, meal_type):
    """Parse and upsert a recipe into Supabase."""
    try:
        # Get full details if not already present
        if "extendedIngredients" not in recipe_data:
            details = fetch_recipe_details(recipe_data["id"])
            if not details:
                return False
            recipe_data = details

        ingredients = parse_ingredients(
            recipe_data.get("extendedIngredients", [])
        )

        # Build instructions string
        instructions = ""
        analyzed = recipe_data.get("analyzedInstructions", [])
        if analyzed and analyzed[0].get("steps"):
            steps = analyzed[0]["steps"]
            instructions = "\n".join(
                f"{s['number']}. {s['step']}" for s in steps
            )
        elif recipe_data.get("instructions"):
            instructions = recipe_data["instructions"]

        # Map dish types to our meal_types
        dish_types = recipe_data.get("dishTypes", [])
        meal_types_mapped = []
        for dt in dish_types:
            dt_lower = dt.lower()
            if any(x in dt_lower for x in ["breakfast", "brunch", "morning"]):
                meal_types_mapped.append("breakfast")
            elif any(x in dt_lower for x in ["lunch", "salad", "soup"]):
                meal_types_mapped.append("lunch")
            elif any(x in dt_lower for x in ["dinner", "main", "course"]):
                meal_types_mapped.append("dinner")
            elif any(x in dt_lower for x in ["snack", "appetizer", "side"]):
                meal_types_mapped.append("snack")
        # Always include the meal type we searched for
        if meal_type not in meal_types_mapped:
            meal_types_mapped.append(meal_type)
        meal_types_mapped = list(set(meal_types_mapped))

        row = {
            "spoonacular_id": recipe_data["id"],
            "name":           recipe_data.get("title", ""),
            "cuisine":        recipe_data.get("cuisines", ["Other"])[0]
                              if recipe_data.get("cuisines") else "Other",
            "meal_types":     meal_types_mapped,
            "prep_time":      recipe_data.get("preparationMinutes") or 0,
            "cook_time":      recipe_data.get("cookingMinutes") or 0,
            "total_time":     recipe_data.get("readyInMinutes") or 0,
            "servings":       recipe_data.get("servings") or 2,
            "ingredients":    ingredients,
            "instructions":   instructions,
            "image_url":      recipe_data.get("image", ""),
            "tags":           recipe_data.get("diets", []),
        }

        # Upsert so re-runs don't create duplicates
        supabase.table("recipes").upsert(
            row, on_conflict="spoonacular_id"
        ).execute()
        return True

    except Exception as e:
        print(f"  Error storing recipe {recipe_data.get('id')}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("KitchenAI — Recipe Fetcher")
    print("=" * 55)

    total_stored = 0
    total_attempts = 0

    for cuisine in CUISINES:
        for meal_type in MEAL_TYPES:
            print(f"\n[{cuisine} / {meal_type}] Fetching...")
            recipes = fetch_recipes_by_cuisine(cuisine, meal_type, num=10)

            if not recipes:
                print(f"  No results.")
                continue

            for recipe in recipes:
                total_attempts += 1
                success = store_recipe(recipe, meal_type)
                if success:
                    total_stored += 1
                    print(f"  ✓ {recipe.get('title', 'Unknown')}")
                time.sleep(0.5)  # be kind to the API

    print(f"\n{'='*55}")
    print(f"✅ Done! Stored {total_stored}/{total_attempts} recipes.")
    print(f"   Check your Supabase recipes table.")
