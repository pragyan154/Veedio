import time
import json
from pathlib import Path
from typing import Callable, Iterable, TypeVar, List, Any, Optional

T = TypeVar("T")

def call_with_models(
    models: Iterable[str],
    fn: Callable[[str], T],
    retries_per_model: int = 5,
    sleep_seconds: int = 5,
) -> T:
    """
    Try fn(model_name) for each model in models in order.
    For each model:
      - up to retries_per_model attempts
      - sleep sleep_seconds between failures
    Returns fn(...) result on first success.
    Raises the last exception if all models fail.
    """
    last_exc = None
    
    # Ensure models is a list/iterable we can loop over
    if isinstance(models, str):
        models = [models]
        
    for model_name in models:
        for attempt in range(1, retries_per_model + 1):
            try:
                print(f"Attempting with model: {model_name} (Attempt {attempt}/{retries_per_model})")
                return fn(model_name)
            except Exception as e:
                print(f"Error with model {model_name} (Attempt {attempt}): {e}")
                last_exc = e
                if attempt < retries_per_model:
                    time.sleep(sleep_seconds)
                else:
                    print(f"All attempts failed for model {model_name}. Switching to next model...")
                    break  # go to next model
                    
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("call_with_models: no models provided or all failed")


# ============================================================
# Provider-aware helpers
# ============================================================
def _load_config() -> dict:
    """Load config.json from the same directory."""
    config_path = Path(__file__).resolve().parent / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_ai_provider(task: str = "default") -> str:
    """Return the configured AI provider for a specific task: 'gemini' or 'openrouter'."""
    cfg = _load_config()
    providers = cfg.get("ai_providers", {})
    return providers.get(task, cfg.get("ai_provider", "gemini")).lower()


def smart_text_call(prompt: str, image=None, task: str = "default") -> str:
    """
    Provider-aware text call.
    If provider is 'openrouter', uses OpenRouter; otherwise Gemini.
    `image` is only supported with Gemini (PIL Image or path str) or OpenRouter vision (path str).
    """
    provider = get_ai_provider(task)
    cfg = _load_config()

    if provider == "openrouter":
        from openrouter_utils import openrouter_text_call, openrouter_vision_call
        models = cfg.get("openrouter_text_models", [])
        if isinstance(models, str):
            models = [models]

        if image is not None and isinstance(image, str):
            # image is a file path → vision call
            def _call(model_name: str):
                return openrouter_vision_call(model_name, prompt, image)
        else:
            def _call(model_name: str):
                return openrouter_text_call(model_name, prompt)

        return call_with_models(
            models,
            _call,
            retries_per_model=cfg.get("max_retries", 5),
            sleep_seconds=int(cfg.get("base_delay", 2)),
        )
    else:
        # Gemini path — import lazily to keep module independent
        import os
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        models = cfg.get("text_models", ["gemini-2.5-flash"])
        if isinstance(models, str):
            models = [models]

        def _call(model_name: str):
            if image is not None:
                # If image is a path string, open it for Gemini, else assume it's PIL or Part
                if isinstance(image, str):
                    from PIL import Image
                    with Image.open(image) as img:
                        contents = [prompt, img]
                        res = client.models.generate_content(model=model_name, contents=contents)
                else:
                    contents = [prompt, image]
                    res = client.models.generate_content(model=model_name, contents=contents)
            else:
                contents = [prompt]
                res = client.models.generate_content(model=model_name, contents=contents)
            return res.text.strip() if res and res.text else ""

        return call_with_models(
            models,
            _call,
            retries_per_model=cfg.get("max_retries", 5),
            sleep_seconds=int(cfg.get("base_delay", 2)),
        )
