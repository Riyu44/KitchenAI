"""
fridge_scanner.py
Fridge photo → detected ingredients → inventory update

Pipeline:
  1. YOLO detects all food items in the photo (bounding boxes)
  2. EfficientNet classifies each detected crop (known items)
  3. CLIP zero-shot handles low-confidence items (unknown items)
  4. Returns structured list of detected ingredients

Usage (standalone test):
  python fridge_scanner.py --image path/to/fridge.jpg
"""

import os
import io
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import clip                     # pip install git+https://github.com/openai/CLIP.git
from ultralytics import YOLO    # pip install ultralytics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
WEIGHTS_DIR     = Path(__file__).parent / "weights"
EFFICIENTNET_PT = WEIGHTS_DIR / "efficientnet_food_best.pt"
CLASS_LABELS_JSON = WEIGHTS_DIR / "class_labels.json"

# Confidence thresholds
YOLO_CONF_THRESHOLD   = 0.30   # minimum YOLO detection confidence
EFFNET_CONF_THRESHOLD = 0.60   # below this → hand off to CLIP
CLIP_CONF_THRESHOLD   = 0.20   # below this → label as "unknown"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 300   # EfficientNet-B3 input size

# Common kitchen ingredients for CLIP zero-shot
# These are the text labels CLIP compares against
CLIP_INGREDIENT_LABELS = [
    # Vegetables
    "tomato", "onion", "potato", "carrot", "capsicum", "bell pepper",
    "cucumber", "spinach", "broccoli", "cauliflower", "cabbage",
    "peas", "corn", "garlic", "ginger", "chilli", "green chilli",
    "bitter gourd", "bottle gourd", "eggplant", "brinjal", "okra",
    "lady finger", "radish", "beetroot", "mushroom", "pumpkin",
    # Fruits
    "apple", "banana", "mango", "orange", "lemon", "lime",
    "grapes", "strawberry", "watermelon", "pineapple", "papaya",
    # Dairy & proteins
    "milk", "yogurt", "curd", "paneer", "cheese", "butter", "ghee",
    "eggs", "chicken", "mutton", "fish", "tofu",
    # Grains & staples
    "rice", "bread", "roti", "chapati", "flour", "wheat",
    # Packaged
    "milk carton", "juice bottle", "sauce bottle", "oil bottle",
    "jam", "pickle jar", "water bottle",
    # Cooked food
    "dal", "curry", "biryani", "sabzi", "rice dish",
]


# ── Model loader ──────────────────────────────────────────────────────────────
class FridgeScanner:
    """
    Full fridge scanning pipeline.
    Loads all three models once and keeps them in memory.
    """

    def __init__(self):
        self.yolo_model    = None
        self.effnet_model  = None
        self.clip_model    = None
        self.clip_preprocess = None
        self.class_labels  = {}
        self.clip_text_features = None
        self._load_models()

    def _load_models(self):
        logger.info("Loading fridge scanner models...")
        t0 = time.time()

        # 1. YOLO — pretrained YOLOv8n (nano, fast) for food detection
        #    We use the pretrained COCO model — it detects 80 object classes
        #    including many food items. Good enough for fridge scanning.
        logger.info("  Loading YOLO...")
        self.yolo_model = YOLO("yolov8n.pt")   # auto-downloads ~6MB

        # 2. EfficientNet — our fine-tuned food classifier
        logger.info("  Loading EfficientNet...")
        if not EFFICIENTNET_PT.exists():
            raise FileNotFoundError(
                f"EfficientNet weights not found at {EFFICIENTNET_PT}. "
                "Run the Kaggle training notebook first and download the weights."
            )

        checkpoint = torch.load(EFFICIENTNET_PT, map_location=DEVICE)
        self.class_labels = checkpoint.get("class_labels", {})
        num_classes = checkpoint.get("num_classes", len(self.class_labels))

        # Also load from JSON if checkpoint doesn't have labels
        if not self.class_labels and CLASS_LABELS_JSON.exists():
            with open(CLASS_LABELS_JSON) as f:
                self.class_labels = json.load(f)
            num_classes = len(self.class_labels)

        self.effnet_model = timm.create_model(
            "efficientnet_b3", pretrained=False, num_classes=num_classes
        ).to(DEVICE)
        self.effnet_model.load_state_dict(checkpoint["model_state"])
        self.effnet_model.eval()

        # 3. CLIP — for zero-shot fallback on unknown items
        logger.info("  Loading CLIP...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
        self.clip_model.eval()

        # Pre-encode all ingredient text labels (done once, reused for every image)
        logger.info("  Pre-encoding CLIP text features...")
        text_tokens = clip.tokenize(
            [f"a photo of {label}" for label in CLIP_INGREDIENT_LABELS]
        ).to(DEVICE)
        with torch.no_grad():
            self.clip_text_features = self.clip_model.encode_text(text_tokens)
            self.clip_text_features = F.normalize(self.clip_text_features, dim=-1)

        logger.info(f"  All models loaded in {time.time()-t0:.1f}s on {DEVICE}")

    # ── Image preprocessing ───────────────────────────────────────────────────
    @property
    def effnet_transform(self):
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    # ── Step 1: YOLO detection ────────────────────────────────────────────────
    def detect_items(self, image: Image.Image) -> list:
        """
        Run YOLO on the full image.
        Returns list of bounding boxes for detected food items.
        Each box: {xyxy, confidence, class_name, crop}
        """
        results = self.yolo_model(
            image,
            conf=YOLO_CONF_THRESHOLD,
            verbose=False
        )[0]

        detections = []
        img_w, img_h = image.size

        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.yolo_model.names[cls_id]

            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))

            # Skip tiny boxes (likely noise)
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w < 30 or box_h < 30:
                continue

            # Add padding around crop for better classification
            pad = 10
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(img_w, x2 + pad)
            y2p = min(img_h, y2 + pad)

            crop = image.crop((x1p, y1p, x2p, y2p))

            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "yolo_conf":  conf,
                "yolo_class": cls_name,
                "crop":       crop,
            })

        logger.info(f"  YOLO detected {len(detections)} items")
        return detections

    # ── Step 2: EfficientNet classification ───────────────────────────────────
    def classify_crop(self, crop: Image.Image) -> tuple[str, float]:
        """
        Classify a single cropped item with EfficientNet.
        Returns (class_name, confidence).
        """
        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        tensor = self.effnet_transform(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.effnet_model(tensor)
            probs  = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)

        conf      = float(conf[0])
        class_idx = str(int(idx[0]))
        class_name = self.class_labels.get(class_idx, f"class_{class_idx}")
        # Clean up class name
        class_name = class_name.replace("_", " ").lower().strip()

        return class_name, conf

    # ── Step 3: CLIP zero-shot ─────────────────────────────────────────────────
    def clip_classify(self, crop: Image.Image) -> tuple[str, float]:
        """
        Zero-shot classification with CLIP.
        Used when EfficientNet confidence is below threshold.
        Returns (ingredient_name, confidence).
        """
        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        clip_input = self.clip_preprocess(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(clip_input)
            image_features = F.normalize(image_features, dim=-1)

            # Cosine similarity between image and all text labels
            similarity = (image_features @ self.clip_text_features.T).squeeze(0)
            probs      = F.softmax(similarity * 100, dim=0)  # temperature scaling
            conf, idx  = probs.max(dim=0)

        conf       = float(conf)
        label      = CLIP_INGREDIENT_LABELS[int(idx)]

        return label, conf

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def scan(self, image: Image.Image) -> dict:
        """
        Full fridge scan pipeline.
        Input:  PIL Image
        Output: {detected_items, annotated_image, scan_stats}
        """
        t0 = time.time()
        logger.info(f"Scanning image {image.size}...")

        # Resize large images for speed (keep aspect ratio)
        max_dim = 1024
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Step 1: Detect all items
        detections = self.detect_items(image)

        # If YOLO finds nothing (e.g. single ingredient photo),
        # treat the whole image as one item
        if not detections:
            logger.info("  No YOLO detections — classifying full image as single item")
            detections = [{
                "bbox":       [0, 0, image.width, image.height],
                "yolo_conf":  1.0,
                "yolo_class": "unknown",
                "crop":       image,
            }]

        # Step 2+3: Classify each detection
        detected_items = []
        stats = {"effnet_used": 0, "clip_used": 0, "unknown": 0}

        for det in detections:
            crop = det["crop"]

            # Try EfficientNet first
            effnet_name, effnet_conf = self.classify_crop(crop)

            if effnet_conf >= EFFNET_CONF_THRESHOLD:
                # EfficientNet confident — use its prediction
                final_name = effnet_name
                final_conf = effnet_conf
                source     = "efficientnet"
                stats["effnet_used"] += 1
            else:
                # EfficientNet uncertain — try CLIP
                clip_name, clip_conf = self.clip_classify(crop)

                if clip_conf >= CLIP_CONF_THRESHOLD:
                    final_name = clip_name
                    final_conf = clip_conf
                    source     = "clip_zero_shot"
                    stats["clip_used"] += 1
                else:
                    # Both models uncertain — mark as unknown
                    final_name = "unknown item"
                    final_conf = max(effnet_conf, clip_conf)
                    source     = "unknown"
                    stats["unknown"] += 1

            detected_items.append({
                "ingredient":    final_name,
                "confidence":    round(final_conf * 100, 1),
                "source":        source,
                "bbox":          det["bbox"],
                "yolo_class":    det["yolo_class"],
                "yolo_conf":     round(det["yolo_conf"] * 100, 1),
                # For inventory: suggest standard unit
                "suggested_qty":  1,
                "suggested_unit": self._suggest_unit(final_name),
            })

        # Step 4: Deduplicate — merge same ingredient detected multiple times
        detected_items = self._deduplicate(detected_items)

        # Step 5: Annotate image for display
        annotated = self._annotate_image(image.copy(), detected_items)

        scan_time = round(time.time() - t0, 2)
        logger.info(f"  Scan complete in {scan_time}s | "
                    f"Found {len(detected_items)} unique ingredients | "
                    f"EfficientNet: {stats['effnet_used']} | "
                    f"CLIP: {stats['clip_used']} | "
                    f"Unknown: {stats['unknown']}")

        return {
            "detected_items":  detected_items,
            "item_count":      len(detected_items),
            "scan_time_sec":   scan_time,
            "stats":           stats,
            "annotated_image": annotated,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _suggest_unit(self, ingredient_name: str) -> str:
        """Suggest a sensible default unit for common ingredients."""
        name = ingredient_name.lower()
        if any(w in name for w in ["milk", "juice", "oil", "ghee", "water"]):
            return "litre"
        if any(w in name for w in ["flour", "rice", "sugar", "salt", "dal", "lentil"]):
            return "kg"
        if any(w in name for w in ["egg"]):
            return "piece"
        if any(w in name for w in ["chicken", "mutton", "fish", "meat", "paneer"]):
            return "g"
        return "piece"   # default for vegetables, fruits

    def _deduplicate(self, items: list) -> list:
        """
        Merge detections of the same ingredient.
        If the same ingredient appears 3 times, count = 3 pieces.
        """
        seen = {}
        for item in items:
            key = item["ingredient"]
            if key in seen:
                seen[key]["suggested_qty"] += 1
                seen[key]["confidence"] = max(
                    seen[key]["confidence"], item["confidence"]
                )
            else:
                seen[key] = dict(item)
        return list(seen.values())

    def _annotate_image(self, image: Image.Image, items: list) -> Image.Image:
        """Draw bounding boxes and labels on the image."""
        draw = ImageDraw.Draw(image)

        color_map = {
            "efficientnet": "#4ade80",   # green
            "clip_zero_shot": "#facc15", # yellow
            "unknown": "#f87171",        # red
        }

        for item in items:
            bbox   = item.get("bbox", [])
            source = item.get("source", "unknown")
            color  = color_map.get(source, "#888888")

            if len(bbox) == 4:
                draw.rectangle(bbox, outline=color, width=3)
                label = f"{item['ingredient']} ({item['confidence']}%)"
                draw.text(
                    (bbox[0] + 4, bbox[1] + 4),
                    label,
                    fill=color,
                )

        return image


# ── FastAPI integration ───────────────────────────────────────────────────────
# This is imported by main.py — scanner is initialised once at startup

_scanner_instance: Optional[FridgeScanner] = None

def get_scanner() -> FridgeScanner:
    """Singleton — load models once, reuse for every request."""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = FridgeScanner()
    return _scanner_instance


def scan_image_bytes(image_bytes: bytes) -> dict:
    """
    Entry point called by FastAPI /scan endpoint.
    Input:  raw image bytes (from uploaded file)
    Output: scan result dict
    """
    image = Image.open(io.BytesIO(image_bytes))
    scanner = get_scanner()
    result  = scanner.scan(image)

    # Remove PIL image from result (not JSON serialisable)
    result.pop("annotated_image", None)

    return result


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fridge scanner on an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--save",  type=str, default="annotated.jpg",
                        help="Save annotated image to this path")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        exit(1)

    print(f"Scanning: {image_path}")
    scanner = FridgeScanner()
    image   = Image.open(image_path)
    result  = scanner.scan(image)

    print(f"\n{'='*50}")
    print(f"Detected {result['item_count']} ingredients in {result['scan_time_sec']}s:")
    print(f"{'='*50}")
    for item in result["detected_items"]:
        source_tag = {"efficientnet": "🟢", "clip_zero_shot": "🟡", "unknown": "🔴"}
        tag = source_tag.get(item["source"], "⚪")
        print(f"  {tag} {item['ingredient']:<25} "
              f"{item['confidence']}% conf | "
              f"{item['suggested_qty']} {item['suggested_unit']} | "
              f"via {item['source']}")

    print(f"\nStats: {result['stats']}")

    # Save annotated image
    result_with_image = scanner.scan(image)
    result_with_image["annotated_image"].save(args.save)
    print(f"\nAnnotated image saved to: {args.save}")
