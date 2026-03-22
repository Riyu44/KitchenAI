# KitchenAI — Claude Context File
> Paste this file into Claude at the start of any new conversation to restore full project context instantly.
> Keep the "Current Status" and "Decisions Log" sections updated as you build.

---

## 👥 Team

| Person | Role | Responsibilities |
|---|---|---|
| Priyansh Shrivastava | ML / Data Science | FastAPI backend, all ML models, recommendation engine, fridge scanner (CV), LLM fine-tuning, LSTM forecasting, inventory optimiser, HuggingFace deployment |
| [Friend's name] | Software Developer | React Native app, Supabase DB schema, Zepto/Blinkit API integration, auth, push notifications, Play Store deployment |

---

## 🎯 Product Vision

**Problem:** "The cook arrives and nobody knows what to make."

**Solution:** An Android app that knows what's in your fridge/pantry (via Zepto purchase history + manual input + fridge photo scan), and recommends what to cook for breakfast/lunch/dinner — accounting for number of people, preferences, and what ingredients are available.

**Core user flow:**
1. App knows current inventory (via Zepto API + manual input + fridge photo)
2. User says: meal time + number of people (+ optional preferences)
3. App recommends ranked recipes based on available ingredients
4. User picks a recipe → app calculates missing ingredients → one-tap Zepto order
5. Post-meal: inventory auto-updated (or manual override)
6. Over time: app learns household preferences and consumption patterns

---

## ✨ Core Features

### 1. Smart Inventory Management
- Zepto/Blinkit API integration (or order history email parsing) for auto-inventory update
- Manual add for offline purchases
- Manual consumption override post-meal
- Expiry tracking with alerts
- Supabase (PostgreSQL) as the inventory database

### 2. AI Recipe Recommendation Engine
- Ranked recipe suggestions based on: ingredient match %, number of people, meal time, cuisine preference, past ratings
- Multi-layer ML: KNN content-based + SVD collaborative filtering
- Ingredient embedding space (sentence-transformers)

### 3. Smart Grocery Ordering
- User picks recipe → app calculates missing ingredients → quantity adjusted for serving size × number of people → one-tap Zepto order

### 4. Fridge Vision (CV flagship feature)
- Take photo of fridge/vegetables → EfficientNet + CLIP identifies items → inventory auto-updated
- Zero-shot detection via CLIP for unseen ingredients

### 5. Natural Language Interface
- "What can I make for 4 people with paneer and spinach in under 30 mins?"
- QLoRA fine-tuned Phi-2 / Mistral-7B interprets intent → filters inventory → returns ranked suggestions

### 6. Consumption Pattern Learning
- LSTM forecasting of weekly ingredient consumption
- Auto-suggest grocery orders before stock runs out
- Contextual bandits for personalisation from user feedback

---

## 🧠 ML Pipeline (Priyansh's work)

### Model 1 — Recipe Recommender (KNN + SVD)
- **Content-based:** Embed recipes as vectors (ingredients, cuisine, cook time) via sentence-transformers → KNN finds recipes matching current inventory
- **Collaborative:** SVD matrix factorisation on user-recipe ratings → latent taste preferences
- **Ensemble:** KNN 50% + SVD 30% + ingredient match % 20%
- **Library:** scikit-learn, sentence-transformers

### Model 2 — Fridge Scanner (EfficientNet + CLIP)
- **EfficientNet-B3** fine-tuned on Food-101 + custom Indian vegetable dataset (Kaggle GPU)
- **CLIP** for zero-shot detection of items not in training set
- **YOLO** optional for multi-item detection in a single fridge photo
- **Dataset:** Food-101 (Kaggle), custom vegetable images

### Model 3 — Natural Language Query (QLoRA fine-tuned LLM)
- **Base model:** Phi-2 (2.7B) or Mistral-7B-Instruct
- **Fine-tuning:** QLoRA (4-bit quantisation + LoRA adapters via PEFT library)
- **Dataset:** Recipe Q&A pairs + instruction-following examples
- **Training:** Kaggle T4 GPU (~2-3 hours)
- **Chain:** User query → fine-tuned LLM → structured intent → inventory filter → ranked recipes

### Model 4 — Consumption Forecaster (LSTM)
- **Architecture:** Multi-variate LSTM in PyTorch/TensorFlow
- **Inputs:** Day of week, number of people, meal type, season, historical consumption
- **Output:** Predicted ingredient consumption for next 7 days → auto grocery list
- **Baseline comparison:** Holt-Winters (already built by Priyansh at Indus Insights)

### Model 5 — Inventory Optimiser (Linear Programming)
- **Library:** PuLP
- **Objective:** Maximise taste preference + minimise food waste + minimise grocery cost
- **Constraints:** Available inventory, expiry dates, dietary restrictions, serving size

### Model 6 — Ingredient NER (DistilBERT / spaCy)
- Parse unstructured recipe text ("2 medium onions, finely chopped") into structured units
- Named Entity Recognition: ingredient name + quantity + unit
- Fine-tune on recipe corpus or use spaCy's en_core_web model

---

## 🏗️ System Architecture

```
📱 React Native App (Expo)
        ↓
FastAPI Backend (HuggingFace Spaces — Docker)
        ↓
┌───────────────────────────────────────────┐
│  ML Engine                                │
│  ├── Recipe Recommender (KNN + SVD)       │
│  ├── Fridge Scanner (EfficientNet + CLIP) │
│  ├── NL Query (Fine-tuned LLM)            │
│  ├── Consumption Forecaster (LSTM)        │
│  └── Inventory Optimiser (PuLP)           │
└───────────────────────────────────────────┘
        ↓
Supabase (PostgreSQL)          Zepto/Blinkit API
├── users                      └── purchase history
├── inventory                      → auto inventory sync
├── recipes
├── meal_history
└── user_preferences
```

---

## 🛠️ Full Tech Stack

| Layer | Technology |
|---|---|
| Mobile frontend | React Native + Expo |
| Backend API | FastAPI (Python) |
| ML framework | PyTorch + TensorFlow + scikit-learn |
| LLM fine-tuning | HuggingFace PEFT + QLoRA |
| CV models | EfficientNet-B3 + CLIP (torchvision + openai/clip) |
| NLP | DistilBERT + sentence-transformers + spaCy |
| Time series | LSTM (PyTorch) |
| Optimisation | PuLP (linear programming) |
| Database | Supabase (PostgreSQL + real-time) |
| Recipe data | Spoonacular API (free tier, 150 req/day) |
| Backend hosting | HuggingFace Spaces (Docker, always-on free) |
| Mobile build | EAS Build (Expo) |
| Project page | GitHub Pages |
| Training compute | Kaggle Notebooks (T4 GPU, free) |

---

## 📅 4-Week Build Plan

### Week 1 — Foundation (Both)
- **Priyansh:** FastAPI skeleton + Supabase schema + Spoonacular recipe fetch + ingredient embedding + basic KNN recommender
- **Friend:** React Native project setup + home screen + inventory input screen + navigation

### Week 2 — CV + Core Recommendation (Priyansh)
- Fine-tune EfficientNet-B3 on Food-101 (Kaggle GPU)
- CLIP zero-shot integration for unknown ingredients
- Fridge photo → inventory update endpoint
- SVD collaborative filter as second rec layer

### Week 3 — LLM + NER (Priyansh) / App screens (Friend)
- **Priyansh:** QLoRA fine-tune Phi-2 on recipe Q&A dataset + ingredient NER (spaCy)
- **Friend:** Recipe detail screen + preferences screen + fridge camera screen

### Week 4 — Forecasting + Integration (Both)
- **Priyansh:** LSTM consumption forecaster + PuLP inventory optimiser + Zepto integration
- **Friend:** Grocery ordering flow + meal history screen + post-meal inventory update
- **Both:** End-to-end testing + HuggingFace deploy + GitHub Pages landing page

---

## 🗄️ Database Schema (Supabase)

```sql
-- Users
users: id, name, household_size, dietary_restrictions[], cuisine_preferences[]

-- Inventory
inventory: id, user_id, ingredient_name, quantity, unit, expiry_date, 
           last_updated, source (zepto/manual/fridge_scan)

-- Recipes
recipes: id, name, cuisine, meal_type[], prep_time, cook_time, servings,
         ingredients (JSONB), instructions, image_url, spoonacular_id

-- Meal History
meal_history: id, user_id, recipe_id, cooked_at, num_people, rating, notes

-- User Preferences
user_preferences: id, user_id, ingredient_id, preference_score, 
                  updated_at (updated by contextual bandit)
```

---

## 🔑 API Keys Needed

| Service | Where to get | Cost |
|---|---|---|
| Spoonacular API | spoonacular.com/food-api | Free (150 req/day) |
| Supabase | supabase.com | Free tier |
| Zepto/Blinkit | No public API yet — use email parsing or manual input workaround | Free |
| HuggingFace | huggingface.co | Free |
| Kaggle | kaggle.com (for GPU training) | Free |

---

## 📁 Repo Structure

```
kitchenai/
├── backend/                    # Priyansh
│   ├── main.py                 # FastAPI app
│   ├── models/
│   │   ├── recommender.py      # KNN + SVD
│   │   ├── fridge_scanner.py   # EfficientNet + CLIP
│   │   ├── nlq.py              # Fine-tuned LLM
│   │   ├── forecaster.py       # LSTM
│   │   └── optimiser.py        # PuLP
│   ├── data/
│   │   ├── fetch_recipes.py    # Spoonacular fetch
│   │   └── build_embeddings.py # Ingredient embeddings
│   ├── training/               # Kaggle notebooks
│   │   ├── train_efficientnet.ipynb
│   │   └── finetune_llm.ipynb
│   ├── requirements.txt
│   └── Dockerfile
├── mobile/                     # Friend
│   ├── App.js
│   ├── src/
│   │   ├── screens/
│   │   ├── components/
│   │   └── api.js
│   └── package.json
├── docs/                       # GitHub Pages
│   └── index.html
└── CLAUDE_CONTEXT.md           # This file
```

---

## ✅ Decisions Already Made

- Mobile: React Native + Expo (not Flutter) — Priyansh has experience from CineMatch
- Backend: FastAPI on HuggingFace Spaces (not Render — had deployment issues)
- DB: Supabase (not Firebase) — PostgreSQL gives better query flexibility for recipe matching
- CV model: EfficientNet-B3 (not ResNet) — better accuracy/compute tradeoff
- LLM: Phi-2 first (faster to fine-tune), upgrade to Mistral-7B if quality insufficient
- Zepto API: No public API — use email order parsing + manual input as fallback
- Training: Kaggle T4 GPU (not Google Colab) — Priyansh's preferred environment

---

## 🔄 Current Status
> **Update this section every time you make progress**

- [ ] Week 1: Foundation
- [ ] Week 2: CV + Recommendation
- [ ] Week 3: LLM + NER
- [ ] Week 4: Forecasting + Integration
- [ ] Deployed to HuggingFace Spaces
- [ ] GitHub Pages landing page live
- [ ] Play Store submission

**Last updated:** [date]
**Currently working on:** Planning phase — not started

---

## 🐛 Known Issues / Blockers
> Add issues here as they come up

- None yet

---

## 💬 How to use this file with Claude

1. Copy the entire contents of this file
2. Start a new Claude conversation
3. Paste it in and say: *"This is our project context. [Your specific question or task]"*
4. Claude will have full context and can continue development seamlessly

**For Priyansh:** Focus questions on ML models, FastAPI, Python code, training notebooks
**For [Friend]:** Focus questions on React Native screens, Supabase integration, Zepto API, app deployment
