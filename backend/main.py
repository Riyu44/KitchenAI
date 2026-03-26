"""
main.py — KitchenAI FastAPI Backend
Week 1 skeleton: inventory CRUD, recipe search, basic KNN recommendation.
Week 2+ will add: fridge scanner, LLM query, LSTM forecaster.
"""

import os
import json
import pickle
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import date

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Supabase client ───────────────────────────────────────────────────────────
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# ── Global model state (loaded once at startup) ───────────────────────────────
class ModelState:
    embeddings:    np.ndarray   = None
    knn:           object       = None
    recipe_index:  dict         = None
    sentence_model: object      = None
    ready:         bool         = False

state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup."""
    print("KitchenAI API starting up...")
    try:
        state.embeddings    = np.load("weights/recipe_embeddings.npy")
        with open("weights/knn_model.pkl",    "rb") as f: state.knn          = pickle.load(f)
        with open("weights/recipe_index.pkl", "rb") as f: state.recipe_index = pickle.load(f)
        state.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        state.ready = True
        print(f"✅ Models loaded — {len(state.recipe_index)} recipes indexed.")
    except FileNotFoundError:
        print("⚠️  Model weights not found. Run data/build_embeddings.py first.")
        print("    Recommendation endpoints will return 503 until weights exist.")
    yield
    print("KitchenAI API shutting down.")


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KitchenAI API",
    version="1.0.0",
    description="Food recommendation backend — inventory, recipes, ML recommendations.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class InventoryItem(BaseModel):
    user_id:         str
    ingredient_name: str
    quantity:        float
    unit:            str
    expiry_date:     Optional[date] = None
    source:          Optional[str]  = "manual"

class InventoryUpdate(BaseModel):
    quantity:    float
    unit:        Optional[str] = None
    expiry_date: Optional[date] = None

class RecommendRequest(BaseModel):
    user_id:    str
    meal_type:  str                    # breakfast | lunch | dinner | snack
    num_people: int = 2
    max_time:   Optional[int] = None   # total cook+prep time in minutes
    cuisine:    Optional[str] = None   # preferred cuisine
    top_n:      Optional[int] = 5      # always return multiple recommendations

class RecipeRecommendation(BaseModel):
    recipe_id:     str
    name:          str
    cuisine:       str
    meal_types:    List[str]
    total_time:    int
    servings:      int
    image_url:     str
    match_score:   float               # 0-100
    missing_items: List[str]           # ingredients not in current inventory
    available_pct: float               # % of ingredients already in fridge

class RecommendResponse(BaseModel):
    recommendations: List[RecipeRecommendation]
    message:         str
    inventory_count: int

class MealLogRequest(BaseModel):
    user_id:          str
    recipe_id:        str
    num_people:       int
    rating:           Optional[int] = None
    notes:            Optional[str] = None
    ingredients_used: Optional[dict] = None

class NLQueryRequest(BaseModel):
    user_id: str
    query:   str    # "What can I make for 4 people with paneer?"

class UserCreate(BaseModel):
    name:                 str
    household_size:       Optional[int]       = 2
    dietary_restrictions: Optional[List[str]] = []
    cuisine_preferences:  Optional[List[str]] = []


# ── Utility functions ─────────────────────────────────────────────────────────

def get_user_inventory(user_id: str) -> list:
    """Fetch all inventory items for a user."""
    result = supabase.table("inventory")\
        .select("*")\
        .eq("user_id", user_id)\
        .execute()
    return result.data


def get_inventory_names(user_id: str) -> set:
    """Return a set of lowercase ingredient names in the user's inventory."""
    inventory = get_user_inventory(user_id)
    return {item["ingredient_name"].lower() for item in inventory}


def calculate_ingredient_match(
    recipe_ingredients: list,
    available: set
) -> tuple[float, list]:
    """
    Returns (available_pct, missing_items) for a recipe given
    the user's current inventory.
    """
    if not recipe_ingredients:
        return 0.0, []

    if isinstance(recipe_ingredients, str):
        recipe_ingredients = json.loads(recipe_ingredients)

    recipe_names = [
        ing.get("name", "").lower()
        for ing in recipe_ingredients
        if ing.get("name")
    ]

    if not recipe_names:
        return 0.0, []

    missing = [name for name in recipe_names if name not in available]
    available_pct = (len(recipe_names) - len(missing)) / len(recipe_names) * 100

    return round(available_pct, 1), missing


def scale_servings(recipe: dict, num_people: int) -> dict:
    """
    Scale ingredient quantities for the requested number of people.
    Returns a copy of the recipe with adjusted quantities.
    """
    original_servings = recipe.get("servings") or 2
    scale_factor = num_people / original_servings

    ingredients = recipe.get("ingredients", [])
    if isinstance(ingredients, str):
        ingredients = json.loads(ingredients)

    scaled = []
    for ing in ingredients:
        scaled_ing = dict(ing)
        if "quantity" in scaled_ing and scaled_ing["quantity"]:
            scaled_ing["quantity"] = round(
                float(scaled_ing["quantity"]) * scale_factor, 2
            )
        scaled.append(scaled_ing)

    return {**recipe, "ingredients": scaled, "servings": num_people}


# ── Routes ────────────────────────────────────────────────────────────────────

# Health check
@app.get("/")
def root():
    return {
        "status": "KitchenAI API is running",
        "models_ready": state.ready,
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {"status": "ok"}


# ── User endpoints ────────────────────────────────────────────────────────────

@app.post("/users")
def create_user(user: UserCreate):
    """Create a new household user."""
    result = supabase.table("users").insert({
        "name":                 user.name,
        "household_size":       user.household_size,
        "dietary_restrictions": user.dietary_restrictions,
        "cuisine_preferences":  user.cuisine_preferences,
    }).execute()
    return {"user": result.data[0], "message": "User created successfully."}


@app.get("/users/{user_id}")
def get_user(user_id: str):
    """Get user profile."""
    result = supabase.table("users").select("*").eq("id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="User not found.")
    return result.data[0]


# ── Inventory endpoints ───────────────────────────────────────────────────────

@app.get("/inventory/{user_id}")
def get_inventory(user_id: str):
    """Get full inventory for a user."""
    inventory = get_user_inventory(user_id)
    return {
        "inventory": inventory,
        "count": len(inventory),
        "message": f"{len(inventory)} items in inventory."
    }


@app.post("/inventory")
def add_inventory_item(item: InventoryItem):
    """Add or update an inventory item."""
    # Check if item already exists for this user
    existing = supabase.table("inventory")\
        .select("id, quantity")\
        .eq("user_id", item.user_id)\
        .eq("ingredient_name", item.ingredient_name.lower())\
        .execute()

    row = {
        "user_id":         item.user_id,
        "ingredient_name": item.ingredient_name.lower(),
        "quantity":        item.quantity,
        "unit":            item.unit,
        "source":          item.source,
        "last_updated":    "now()",
    }
    if item.expiry_date:
        row["expiry_date"] = str(item.expiry_date)

    if existing.data:
        # Update existing item — add quantities
        new_qty = existing.data[0]["quantity"] + item.quantity
        row["quantity"] = new_qty
        result = supabase.table("inventory")\
            .update(row)\
            .eq("id", existing.data[0]["id"])\
            .execute()
        return {"item": result.data[0], "message": f"Updated {item.ingredient_name}."}
    else:
        result = supabase.table("inventory").insert(row).execute()
        return {"item": result.data[0], "message": f"Added {item.ingredient_name}."}


@app.patch("/inventory/{item_id}")
def update_inventory_item(item_id: str, update: InventoryUpdate):
    """Update quantity/unit/expiry of a specific inventory item."""
    row = {"quantity": update.quantity, "last_updated": "now()"}
    if update.unit:        row["unit"] = update.unit
    if update.expiry_date: row["expiry_date"] = str(update.expiry_date)

    result = supabase.table("inventory")\
        .update(row)\
        .eq("id", item_id)\
        .execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Inventory item not found.")
    return {"item": result.data[0], "message": "Inventory updated."}


@app.delete("/inventory/{item_id}")
def delete_inventory_item(item_id: str):
    """Remove an item from inventory."""
    supabase.table("inventory").delete().eq("id", item_id).execute()
    return {"message": "Item removed from inventory."}


@app.post("/inventory/{user_id}/consume")
def consume_ingredients(user_id: str, consumption: dict):
    """
    Post-meal inventory update.
    Body: {"paneer": {"quantity": 200, "unit": "g"}, "spinach": {...}}
    Deducts consumed quantities from inventory.
    """
    updates = []
    for ingredient_name, consumed in consumption.items():
        existing = supabase.table("inventory")\
            .select("id, quantity")\
            .eq("user_id", user_id)\
            .eq("ingredient_name", ingredient_name.lower())\
            .execute()

        if existing.data:
            current_qty = existing.data[0]["quantity"]
            consumed_qty = float(consumed.get("quantity", 0))
            new_qty = max(0, current_qty - consumed_qty)

            supabase.table("inventory")\
                .update({"quantity": new_qty, "last_updated": "now()"})\
                .eq("id", existing.data[0]["id"])\
                .execute()
            updates.append(f"{ingredient_name}: {current_qty} → {new_qty}")

    return {
        "message": f"Inventory updated for {len(updates)} ingredients.",
        "updates": updates
    }


# ── Recipe endpoints ──────────────────────────────────────────────────────────

@app.get("/recipes/search")
def search_recipes(q: str = "", cuisine: str = "", meal_type: str = "", limit: int = 10):
    """Search recipes by name, cuisine, or meal type."""
    query = supabase.table("recipes").select(
        "id, name, cuisine, meal_types, total_time, servings, image_url, tags"
    )
    if q:
        query = query.ilike("name", f"%{q}%")
    if cuisine:
        query = query.eq("cuisine", cuisine)

    result = query.limit(limit).execute()
    recipes = result.data

    # Filter by meal_type in Python (Supabase array contains)
    if meal_type:
        recipes = [r for r in recipes if meal_type in (r.get("meal_types") or [])]

    return {"recipes": recipes, "count": len(recipes)}


@app.get("/recipes/{recipe_id}")
def get_recipe(recipe_id: str, num_people: int = 2):
    """Get full recipe details, scaled to num_people."""
    result = supabase.table("recipes").select("*").eq("id", recipe_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Recipe not found.")
    recipe = result.data[0]
    return scale_servings(recipe, num_people)


# ── Recommendation endpoint ───────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(req: RecommendRequest):
    """
    Main recommendation endpoint.
    Returns top_n recipes ranked by:
      - Ingredient availability (how much of the fridge is already used)
      - Semantic similarity to meal type + cuisine preference
      - Time constraints

    Body example:
    {
      "user_id": "uuid",
      "meal_type": "dinner",
      "num_people": 4,
      "max_time": 60,
      "cuisine": "Indian",
      "top_n": 5
    }
    """
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. Run data/build_embeddings.py first."
        )

    # 1. Get user's inventory
    available_ingredients = get_inventory_names(req.user_id)

    # 2. Build query embedding
    query_text = (
        f"Recipe for {req.meal_type}. "
        f"Cuisine: {req.cuisine or 'any'}. "
        f"Ready in {req.max_time or 60} minutes. "
        f"Serves {req.num_people} people. "
        f"Ingredients available: {', '.join(list(available_ingredients)[:20])}."
    )
    query_embedding = state.sentence_model.encode([query_text])

    # 3. KNN search — get top 50 candidates
    n_candidates = min(50, len(state.recipe_index))
    distances, indices = state.knn.kneighbors(query_embedding, n_neighbors=n_candidates)

    # 4. Score and filter candidates
    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        recipe = state.recipe_index[idx]

        # Filter by meal type
        if req.meal_type and req.meal_type not in (recipe.get("meal_types") or []):
            continue

        # Filter by max time
        if req.max_time and recipe.get("total_time", 0) > req.max_time:
            continue

        # Filter by cuisine preference
        if req.cuisine and recipe.get("cuisine", "").lower() != req.cuisine.lower():
            continue

        # Calculate ingredient availability
        ingredients = recipe.get("ingredients", [])
        if isinstance(ingredients, str):
            ingredients = json.loads(ingredients)

        available_pct, missing = calculate_ingredient_match(
            ingredients, available_ingredients
        )

        # Composite score: 60% ingredient availability + 40% semantic similarity
        semantic_score  = float(1 - dist) * 100   # cosine distance → similarity
        composite_score = (0.6 * available_pct) + (0.4 * semantic_score)

        candidates.append({
            "recipe":        recipe,
            "score":         composite_score,
            "available_pct": available_pct,
            "missing":       missing,
        })

    # 5. Sort by composite score, take top_n
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:req.top_n]

    # 6. Handle empty results — relax filters
    if not top:
        # Relax cuisine filter and retry
        for dist, idx in zip(distances[0], indices[0]):
            recipe = state.recipe_index[idx]
            if req.meal_type and req.meal_type not in (recipe.get("meal_types") or []):
                continue
            ingredients = recipe.get("ingredients", [])
            if isinstance(ingredients, str):
                ingredients = json.loads(ingredients)
            available_pct, missing = calculate_ingredient_match(
                ingredients, available_ingredients
            )
            semantic_score  = float(1 - dist) * 100
            composite_score = (0.6 * available_pct) + (0.4 * semantic_score)
            candidates.append({
                "recipe": recipe, "score": composite_score,
                "available_pct": available_pct, "missing": missing,
            })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:req.top_n]

    # 7. Build response
    recommendations = []
    for c in top:
        r = c["recipe"]
        recommendations.append(RecipeRecommendation(
            recipe_id     = r["id"],
            name          = r["name"],
            cuisine       = r.get("cuisine", ""),
            meal_types    = r.get("meal_types", []),
            total_time    = r.get("total_time", 0),
            servings      = r.get("servings", 2),
            image_url     = r.get("image_url", ""),
            match_score   = round(c["score"], 1),
            missing_items = c["missing"][:5],   # cap at 5 for UI
            available_pct = c["available_pct"],
        ))

    msg = f"Found {len(recommendations)} recipe recommendations"
    if available_ingredients:
        msg += f" using {len(available_ingredients)} items from your inventory."
    else:
        msg += ". Add items to your inventory for personalised results."

    return RecommendResponse(
        recommendations = recommendations,
        message         = msg,
        inventory_count = len(available_ingredients),
    )


# ── Meal logging endpoint ─────────────────────────────────────────────────────

@app.post("/meals/log")
def log_meal(req: MealLogRequest):
    """
    Log a cooked meal. Updates inventory via consumption data.
    Also triggers preference score update (Week 3: contextual bandit).
    """
    row = {
        "user_id":          req.user_id,
        "recipe_id":        req.recipe_id,
        "num_people":       req.num_people,
        "cooked_at":        "now()",
    }
    if req.rating:           row["rating"] = req.rating
    if req.notes:            row["notes"]  = req.notes
    if req.ingredients_used: row["ingredients_used"] = req.ingredients_used

    result = supabase.table("meal_history").insert(row).execute()
    return {
        "meal": result.data[0],
        "message": "Meal logged. Don't forget to update your inventory consumption!"
    }


@app.get("/meals/{user_id}/history")
def get_meal_history(user_id: str, limit: int = 20):
    """Get recent meal history for a user."""
    result = supabase.table("meal_history")\
        .select("*, recipes(name, cuisine, image_url)")\
        .eq("user_id", user_id)\
        .order("cooked_at", desc=True)\
        .limit(limit)\
        .execute()
    return {"history": result.data, "count": len(result.data)}


# ── NL Query endpoint (stub — Week 3 will add LLM) ────────────────────────────

@app.post("/query")
def natural_language_query(req: NLQueryRequest):
    """
    Natural language recipe query.
    Week 1: keyword extraction + search fallback.
    Week 3: replaced with QLoRA fine-tuned LLM.
    """
    query = req.query.lower()

    # Basic keyword extraction for meal type
    meal_type = "dinner"
    if any(w in query for w in ["breakfast", "morning", "brunch"]):
        meal_type = "breakfast"
    elif any(w in query for w in ["lunch", "noon", "midday"]):
        meal_type = "lunch"
    elif any(w in query for w in ["snack", "quick", "light"]):
        meal_type = "snack"

    # Basic cuisine extraction
    cuisine = None
    for c in ["indian", "chinese", "italian", "mexican", "thai"]:
        if c in query:
            cuisine = c.capitalize()
            break

    # Number of people extraction
    num_people = 2
    for word, num in [("one", 1), ("two", 2), ("three", 3), ("four", 4),
                      ("five", 5), ("six", 6), ("1", 1), ("2", 2),
                      ("3", 3), ("4", 4), ("5", 5), ("6", 6)]:
        if word in query:
            num_people = num
            break

    # Delegate to recommend endpoint
    rec_result = get_recommendations(RecommendRequest(
        user_id    = req.user_id,
        meal_type  = meal_type,
        num_people = num_people,
        cuisine    = cuisine,
        top_n      = 5,
    ))

    return {
        **rec_result.dict(),
        "parsed_intent": {
            "meal_type":  meal_type,
            "num_people": num_people,
            "cuisine":    cuisine,
            "note":       "Week 1 keyword extraction. Week 3 upgrades this to LLM."
        }
    }


# ── Grocery list endpoint ─────────────────────────────────────────────────────

@app.get("/grocery/{user_id}/list")
def get_grocery_list(user_id: str, recipe_id: str, num_people: int = 2):
    """
    Given a recipe and number of people, returns:
    - What's already in inventory (no need to buy)
    - What's missing (needs to be ordered)
    - Scaled quantities for the number of people
    """
    # Get recipe
    recipe_result = supabase.table("recipes").select("*").eq("id", recipe_id).execute()
    if not recipe_result.data:
        raise HTTPException(status_code=404, detail="Recipe not found.")

    recipe = scale_servings(recipe_result.data[0], num_people)
    available = get_inventory_names(user_id)

    ingredients = recipe.get("ingredients", [])
    if isinstance(ingredients, str):
        ingredients = json.loads(ingredients)

    have = []
    need = []

    for ing in ingredients:
        name = ing.get("name", "").lower()
        if name in available:
            have.append(ing)
        else:
            need.append(ing)

    return {
        "recipe_name":   recipe["name"],
        "num_people":    num_people,
        "have":          have,
        "need":          need,
        "need_count":    len(need),
        "have_count":    len(have),
        "ready_pct":     round(len(have) / len(ingredients) * 100, 1) if ingredients else 0,
        "message":       f"You have {len(have)}/{len(ingredients)} ingredients. "
                         f"Need to order {len(need)} items."
    }


# ── Dev utilities ─────────────────────────────────────────────────────────────

@app.get("/dev/stats")
def dev_stats():
    """Quick stats — useful during development."""
    recipe_count    = supabase.table("recipes").select("id", count="exact").execute()
    user_count      = supabase.table("users").select("id", count="exact").execute()
    inventory_count = supabase.table("inventory").select("id", count="exact").execute()
    meal_count      = supabase.table("meal_history").select("id", count="exact").execute()

    return {
        "recipes":       recipe_count.count,
        "users":         user_count.count,
        "inventory_rows": inventory_count.count,
        "meals_logged":  meal_count.count,
        "models_ready":  state.ready,
    }


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
