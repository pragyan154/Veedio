# openrouter_utils.py
# ============================================================
# OpenRouter API integration for text and image generation.
# Provides the same call pattern as the Gemini-based helpers
# so they can be used interchangeably via ai_utils.
# ============================================================

import os
import json
import time
import base64
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("openrouter")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --------------- low-level request ---------------

def _openrouter_chat(
    model: str,
    messages: list,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """
    Send a chat-completion request to OpenRouter.
    Returns the assistant's text content.
    """
    import httpx  # already in requirements

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/user/newpick",
        "X-Title": "NewPick Pipeline",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {data}")

    return choices[0]["message"]["content"].strip()


# --------------- public helpers ---------------

def openrouter_text_call(model: str, prompt: str) -> str:
    """
    Simple text-in → text-out call via OpenRouter.
    Drop-in replacement for gemini_text_call in many modules.
    """
    messages = [{"role": "user", "content": prompt}]
    return _openrouter_chat(model, messages)


def openrouter_vision_call(model: str, prompt: str, image_path: str) -> str:
    """
    Send an image + text prompt to an OpenRouter vision model.
    The image is base64-encoded inline.
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                    },
                },
            ],
        }
    ]
    return _openrouter_chat(model, messages)


def openrouter_image_generate(model: str, prompt: str, output_path: Path) -> Optional[Path]:
    """
    Generate an image via OpenRouter image-generation endpoint.
    Note: Not all OpenRouter models support image generation.
    Falls back gracefully and returns None if unsupported.
    """
    import httpx

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
    }

    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{OPENROUTER_BASE_URL}/images/generations",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        images = data.get("data", [])
        if not images:
            log.warning("OpenRouter image generation returned no images.")
            return None

        # Download the image URL or decode base64
        img_data = images[0]
        if "b64_json" in img_data:
            img_bytes = base64.b64decode(img_data["b64_json"])
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            return output_path
        elif "url" in img_data:
            img_resp = httpx.get(img_data["url"], timeout=60)
            img_resp.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(img_resp.content)
            return output_path
    except Exception as e:
        log.error(f"OpenRouter image generation failed: {e}")
        return None
