from google import genai
from google.genai import types
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Try importing ai_utils
try:
    from ai_utils import call_with_models
except ImportError:
    try:
        from app.ai_utils import call_with_models
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).resolve().parent))
        from ai_utils import call_with_models

from prompts import DEFAULT_IMAGE_ANALYSIS_PROMPT

def load_config(config_path):
    # Adjust path if needed
    if not os.path.exists(config_path):
        # try relative to this file
        base_dir = Path(__file__).resolve().parent
        config_path = base_dir / config_path
        
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def process_images(IMAGE_DIR):
    load_dotenv()
    # Load config
    config_path = Path(__file__).resolve().parent / "config.json"
    config = load_config(str(config_path))

    OUTPUT_JSON = os.path.join(IMAGE_DIR, "InputImage.json")
    image_descriptions = {}

    for filename in os.listdir(IMAGE_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        image_path = os.path.join(IMAGE_DIR, filename)

        try:
            from ai_utils import smart_text_call
            prompt = config.get("InputImageanalysisprompt", DEFAULT_IMAGE_ANALYSIS_PROMPT)
            description = smart_text_call(prompt, image=image_path, task="image_analysis")
            image_descriptions[filename] = description
        except Exception as e:
            print(f"Failed to process {filename} after all retries: {e}")
            image_descriptions[filename] = ""

    # Save JSON INSIDE the image folder
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(image_descriptions, f, indent=2, ensure_ascii=False)

    return OUTPUT_JSON
