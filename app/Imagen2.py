# image_pipeline.py
# ============================================================
# END-TO-END IMAGE RESOLUTION PIPELINE
# Import-safe, dict-based config, with __main__ entrypoint
# ============================================================

import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError, APIError
from PIL import Image
import re

# Try importing ai_utils from the same directory, or app.ai_utils if running from root
try:
    from ai_utils import call_with_models, get_ai_provider
except ImportError:
    try:
        from app.ai_utils import call_with_models, get_ai_provider
    except ImportError:
        # Last resort: try to find it relative to this file
        import sys
        sys.path.append(str(Path(__file__).resolve().parent))
        from ai_utils import call_with_models, get_ai_provider

from prompts import STRICT_MATCH_PROMPT

# ============================================================
# ENV & CLIENT
# ============================================================
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("image-pipeline")

log.info("Image pipeline initialized")

RETRIABLE_ERRORS = (ServerError, APIError)

def safe_slug(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "", s)
    return s[:max_len] if len(s) > max_len else s\


# ============================================================
# CONFIG LOADER
# ============================================================
def load_config(config_path: str, stage2_file: str, final_mapping_file: str) -> dict:
    log.info(f"Loading config from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["stage2_file"] = stage2_file
    config["final_mapping_file"] = final_mapping_file
    config["image_extensions"] = tuple(config["image_extensions"])

    log.info("Config loaded successfully")
    return config



# ============================================================
# UTILS
# ============================================================
def sleep_backoff(config: dict, attempt: int):
    delay = config["base_delay"] * (2 ** (attempt - 1))
    log.info(f"Sleeping for {delay}s before retry")
    time.sleep(delay)


def list_images(config: dict, folder: Path) -> List[Path]:
    if not folder.exists():
        log.info(f"Image folder does not exist: {folder}")
        return []
    images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in config["image_extensions"]
    ]
    log.info(f"Found {len(images)} images in {folder}")
    return images


def load_progress(config: dict) -> dict:
    if Path(config["tracking_file"]).exists():
        with open(config["tracking_file"], "r", encoding="utf-8") as f:
            data = json.load(f)
            log.info(f"Resuming from index {data.get('last_completed_index', -1)}")
            return {
                "last_completed_index": data.get("last_completed_index", -1)
            }
    log.info("No progress file found, starting fresh")
    return {"last_completed_index": -1}


def save_progress(config: dict, index: int):
    with open(config["tracking_file"], "w", encoding="utf-8") as f:
        json.dump({"last_completed_index": index}, f, indent=2)
    log.info(f"Progress saved at index {index}")


def upsert(mapping_list: list, new_item: dict):
    for i, it in enumerate(mapping_list):
        if it["timeframe"] == new_item["timeframe"]:
            mapping_list[i] = new_item
            log.info(f"Updated mapping for timeframe {new_item['timeframe']}")
            return
    mapping_list.append(new_item)
    log.info(f"Inserted mapping for timeframe {new_item['timeframe']}")

# ============================================================
# GEMINI CALLS
# ============================================================
def gemini_text_call(config: dict, prompt: str, image: Optional[Image.Image] = None) -> str:
    log.info("Calling text model")

    # Try OpenRouter if configured
    provider = get_ai_provider("image_evaluation")
    if provider == "openrouter" and image is None:
        from ai_utils import smart_text_call
        try:
            result = smart_text_call(prompt, task="image_evaluation")
            if result:
                return result
        except Exception as e:
            log.warning(f"OpenRouter text call failed, falling back to Gemini: {e}")

    models = config.get("text_models", [])
    if not models and "text_model" in config:
        models = [config["text_model"]]
    if isinstance(models, str):
        models = [models]

    def _call_text_model(model_name: str):
        log.info(f"Using text model: {model_name}")
        contents = [prompt] if image is None else [prompt, image]
        res = client.models.generate_content(
            model=model_name,
            contents=contents,
        )
        return res.text.strip() if res and res.text else ""

    try:
        return call_with_models(
            models,
            _call_text_model,
            retries_per_model=config.get("max_retries", 5),
            sleep_seconds=config.get("base_delay", 2)
        )
    except Exception as e:
        log.error(f"Gemini text call failed: {e}")
        return ""


def gemini_image_generate(config: dict, description: str, output_path: Path) -> Optional[Path]:
    log.info("Calling image generation")

    # Try OpenRouter image generation if configured
    provider = get_ai_provider("image_generation")
    if provider == "openrouter":
        from openrouter_utils import openrouter_image_generate
        or_models = config.get("openrouter_image_models", [])
        if isinstance(or_models, str):
            or_models = [or_models]
        for or_model in or_models:
            try:
                result = openrouter_image_generate(or_model, description, output_path)
                if result:
                    return result
            except Exception as e:
                log.warning(f"OpenRouter image gen failed with {or_model}: {e}")
        log.info("OpenRouter image generation failed, falling back to Gemini")

    models = config.get("image_model_primary", [])
    if not models and "image_model_fallback" in config:
        models = [config.get("image_model_primary"), config.get("image_model_fallback")]
        models = [m for m in models if m]

    if isinstance(models, str):
        models = [models]

    def _call_image_model(model_name: str):
        log.info(f"Using image model: {model_name}")
        if model_name.startswith("imagen-"):
            res = client.models.generate_images(
                model=model_name,
                prompt=description,
            )
            if res and res.images:
                res.images[0].save(output_path)
                return output_path
        else:
            res = client.models.generate_content(
                model=model_name,
                contents=[description],
                config={"response_modalities": ["IMAGE"]},
            )
            parts = []
            if hasattr(res, "parts") and res.parts:
                parts = res.parts
            elif hasattr(res, "candidates") and res.candidates:
                for c in res.candidates:
                    if c.content and c.content.parts:
                        parts.extend(c.content.parts)

            for part in parts:
                if getattr(part, "inline_data", None):
                    img = part.as_image()
                    img.save(output_path)
                    return output_path

        raise RuntimeError(f"No image generated from model {model_name}")

    try:
        return call_with_models(
            models,
            _call_image_model,
            retries_per_model=config.get("max_retries", 5),
            sleep_seconds=config.get("base_delay", 2)
        )
    except Exception as e:
        log.error(f"Image generation failed: {e}")
        return None

# ============================================================
# IMAGE EVALUATION
# ============================================================
# STRICT_MATCH_PROMPT is now imported from prompts.py


def evaluate_image(config: dict, image_path: Path, subtitle: str, description: str) -> float:
    log.info(f"Evaluating image {image_path}")
    try:
        with Image.open(image_path) as img:
            prompt = STRICT_MATCH_PROMPT.format(
                sub_text=subtitle,
                description=description,
            )
            res = gemini_text_call(config, prompt, img)
            if not res:
                return 0.0

            start = res.find("{")
            end = res.rfind("}") + 1
            if start == -1 or end == 0:
                return 0.0

            data = json.loads(res[start:end])
            return float(data.get("score", 0.0))
    except Exception:
        return 0.0

# ============================================================
# CORE PIPELINE
# ============================================================
def resolve_images(config: dict, stage2_items: list):
    log.info(f"Resolving images for {len(stage2_items)} items")
    progress = load_progress(config)
    start_index = progress.get("last_completed_index", -1) + 1

    final_mapping = []
    if Path(config["final_mapping_file"]).exists():
        with open(config["final_mapping_file"], "r", encoding="utf-8") as f:
            final_mapping = json.load(f)

    existing = {}
    if final_mapping:
        for item in final_mapping:
            if isinstance(item, dict) and item.get("timeframe"):
                existing[item["timeframe"]] = item.get("image_path")

    total = len(stage2_items)

    gemini_out_dir = Path(config["final_mapping_file"]).resolve().parent / "gemini_images"
    gemini_out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Gemini generated images will be stored in: {gemini_out_dir}")

    for idx in range(start_index, total):
        item = stage2_items[idx]
        timeframe = item["timeframe"]

        log.info(f"Processing index {idx} | timeframe {timeframe}")

        if timeframe in existing and existing[timeframe]:
            log.info(f"Skipping timeframe {timeframe}, already resolved")
            continue

        search_queryy = item["search_query"]
        subtitle = item.get("text", "")
        description = item["description"]

        image_path = Path(item["image_path"]) if item.get("image_path") else None
        candidates = list_images(config, image_path.parent) if image_path else []

        best_score = 0.0
        best_image = None

        for img in candidates:
            score = evaluate_image(config, img, search_queryy, description)
            if score > best_score:
                best_score = score
                best_image = img

        if best_image and best_score >= config["strict_match_threshold"]:
            log.info(f"Selected existing image {best_image} with score {best_score}")
            upsert(final_mapping, {
                "timeframe": timeframe,
                "text": subtitle,
                "image_path": str(best_image),
            })
        else:
            log.info("No suitable image found, generating new image")
            slug = safe_slug(timeframe) or f"idx_{idx}"
            gen_path = gemini_out_dir / f"generated_{idx}_{slug}.png"
            generated = gemini_image_generate(config, description, gen_path)
            upsert(final_mapping, {
                "timeframe": timeframe,
                "text": subtitle,
                "image_path": str(generated) if generated else None,
            })

        with open(config["final_mapping_file"], "w", encoding="utf-8") as f:
            json.dump(final_mapping, f, indent=2, ensure_ascii=False)

        save_progress(config, idx)

# ============================================================
# MAIN ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    CONFIG_JSON = "config.json"
    STAGE2_FILE = "/Users/pragyan/Voicecline/app/runs/1769693181452/1769693181452_Download_with_Description.json"
    FINAL_MAPPING_FILE = "/Users/pragyan/Voicecline/app/runs/1769693181452/final_1769693181452.json"

    log.info("Starting image pipeline execution")

    config = load_config(
        config_path=CONFIG_JSON,
        stage2_file=STAGE2_FILE,
        final_mapping_file=FINAL_MAPPING_FILE,
    )

    if not Path(config["stage2_file"]).exists():
        log.error("Stage2 file not found, exiting")
        sys.exit(1)

    with open(config["stage2_file"], "r", encoding="utf-8") as f:
        stage2_data = json.load(f)

    resolve_images(config, stage2_data)

    log.info("Image pipeline completed")
