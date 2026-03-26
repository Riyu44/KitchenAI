"""
build_embeddings.py
Builds sentence embeddings for all recipes in Supabase.
Run after fetch_recipes.py.
Usage: python build_embeddings.py
"""

import os
import json
import pickle
import numpy as np
from supabase import create_client
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


# ── Load recipes from Supabase ────────────────────────────────────────────────
def load_recipes():
    print("Loading recipes from Supabase...")
    result = supabase.table("recipes").select("*").execute()
    recipes = result.data
    print(f"  Loaded {len(recipes)} recipes.")
    return recipes


# ── Build recipe text for embedding ──────────────────────────────────────────
def recipe_to_text(recipe):
    """
    Converts a recipe into a rich text string for embedding.
    The embedding captures: name, cuisine, meal type, ingredients, tags.
    """
    ingredients = recipe.get("ingredients", [])
    if isinstance(ingredients, str):
        ingredients = json.loads(ingredients)

    ingredient_names = [
        ing.get("name", "") for ing in ingredients
        if ing.get("name")
    ]

    parts = [
        f"Recipe: {recipe.get('name', '')}.",
        f"Cuisine: {recipe.get('cuisine', '')}.",
        f"Meal type: {', '.join(recipe.get('meal_types', []))}.",
        f"Ingredients: {', '.join(ingredient_names)}.",
        f"Tags: {', '.join(recipe.get('tags', []))}.",
        f"Ready in {recipe.get('total_time', 0)} minutes.",
    ]
    return " ".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("KitchenAI — Embedding Builder")
    print("=" * 55)

    # 1. Load recipes
    recipes = load_recipes()
    if not recipes:
        print("No recipes found. Run fetch_recipes.py first.")
        exit(1)

    # 2. Build text representations
    print("\nBuilding recipe text representations...")
    recipe_texts = [recipe_to_text(r) for r in recipes]
    recipe_ids   = [r["id"] for r in recipes]
    recipe_names = [r["name"] for r in recipes]

    # 3. Generate embeddings
    print("\nGenerating sentence embeddings (takes 1-2 mins)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        recipe_texts,
        show_progress_bar=True,
        batch_size=32
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # 4. Fit KNN model
    print("\nFitting KNN model...")
    knn = NearestNeighbors(
        n_neighbors=min(20, len(recipes)),
        metric="cosine",
        algorithm="brute"
    )
    knn.fit(embeddings)

    # 5. Save everything
    print("\nSaving models...")
    os.makedirs("weights", exist_ok=True)

    np.save("weights/recipe_embeddings.npy", embeddings)

    with open("weights/knn_model.pkl", "wb") as f:
        pickle.dump(knn, f)

    # Save recipe index: id → name + metadata for fast lookup
    recipe_index = {
        i: {
            "id":         r["id"],
            "name":       r["name"],
            "cuisine":    r["cuisine"],
            "meal_types": r["meal_types"],
            "total_time": r["total_time"],
            "servings":   r["servings"],
            "image_url":  r["image_url"],
            "tags":       r["tags"],
            "ingredients": r["ingredients"],
        }
        for i, r in enumerate(recipes)
    }
    with open("weights/recipe_index.pkl", "wb") as f:
        pickle.dump(recipe_index, f)

    print(f"  Saved {len(recipes)} recipe embeddings.")
    print(f"  Saved KNN model.")
    print(f"  Saved recipe index.")

    # 6. Sanity check
    print("\nSanity check — similar recipes to 'Chicken Tikka Masala':")
    target = next(
        (i for i, r in enumerate(recipes)
         if "tikka" in r["name"].lower()),
        None
    )
    if target is not None:
        distances, indices = knn.kneighbors([embeddings[target]])
        print(f"  Input: {recipe_names[target]}")
        print("  Similar:")
        for idx in indices[0][1:6]:
            print(f"    - {recipe_names[idx]}")
    else:
        print("  Tikka Masala not in dataset — fetch more Indian recipes.")

    print(f"\n{'='*55}")
    print("✅ Embeddings built and saved to weights/")
    print("   Next: run main.py to start the FastAPI server.")
