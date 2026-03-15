# image_srt_mapper.py

import json
import time
import pysrt
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
import os
import sys
from prompts import get_image_mapping_prompt

# ───────────────── CONFIG (CONSTANTS) ─────────────────
MAX_RETRIES = 5
BASE_DELAY = 1.5


def run_mapping(SRT_FILE, INPUT_IMAGE_JSON, OUTPUT_FILE, only_input_image=False):
    # ───────────────── ENV ─────────────────
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # ───────────────── AI CALL ─────────────────
    def gemini_call(prompt: str) -> str:
        print("📡 Sending request to AI via smart_text_call (task: image_mapping)")
        try:
            from ai_utils import smart_text_call
            return smart_text_call(prompt, task="image_mapping")
        except Exception as e:
            print(f"❌ AI call failed: {e}")
            raise e

    # ───────────────── HELPERS ─────────────────
    def build_subtitle_block(subs):
        print(f"📝 Building subtitle block ({len(subs)} lines)")
        return "\n".join(
            f"[{i}] {s.text.replace(chr(10), ' ').strip()}"
            for i, s in enumerate(subs)
        )

    def build_image_block(images):
        print(f"🖼️ Building image block ({len(images)} images)")
        return "\n".join(
            f"- {name}: {desc}"
            for name, desc in images.items()
        )

    # ───────────────── MAIN LOGIC ─────────────────
    def map_images_to_srt(subs, image_inventory, only_input_image=False):
        print(f"🔧 Preparing mapping prompt (Mode: {'ONLY_INPUT' if only_input_image else 'STANDARD'})")
        if not image_inventory:
            print("⚠️ No input images provided. Skipping Gemini mapping.")
            final = []
            for sub in subs:
                final.append({
                    "timeframe": f"{sub.start} --> {sub.end}",
                    "text": sub.text.replace(chr(10), " ").strip(),
                    "search_query": "",
                    "description": "",
                    "image_path": ""
                })
            return final
        subtitle_block = build_subtitle_block(subs)
        image_block = build_image_block(image_inventory)

        prompt = get_image_mapping_prompt(image_block, subtitle_block, only_input_image)

        def _parse_json_from_response(raw_text: str):
            raw_text = (raw_text or "").strip()
            if not raw_text:
                return None
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```", 2)[1].strip()
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()
            # Try to extract a JSON array if extra text is present
            start = raw_text.find("[")
            end = raw_text.rfind("]")
            if start != -1 and end != -1 and end > start:
                raw_text = raw_text[start:end + 1]
            try:
                parsed = json.loads(raw_text)
            except Exception:
                return None
            if not isinstance(parsed, list):
                return None
            return parsed

        assignments = None
        last_raw = ""
        for attempt in range(1, MAX_RETRIES + 1):
            raw = gemini_call(prompt)
            last_raw = raw
            assignments = _parse_json_from_response(raw)
            if assignments is not None:
                break
            wait = BASE_DELAY * (2 ** (attempt - 1))
            print(f"⚠️ JSON parse failed, retrying in {wait:.1f}s")
            time.sleep(wait)

        if assignments is None:
            print("❌ JSON parse failed after retries")
            snippet = (last_raw or "")[:400].replace("\n", " ")
            sys.exit(f"Could not parse JSON. Response snippet: {snippet}")

        index_to_image = {}
        for a in assignments:
            idx = a["subtitle_index"]
            img = a["image"]
            if img not in image_inventory:
                print(f"⚠️ Warning: Invalid image name returned: {img}. Skipping.")
                continue
            index_to_image[idx] = img

        final = []
        for i, sub in enumerate(subs):
            img = index_to_image.get(i)
            final.append({
                "timeframe": f"{sub.start} --> {sub.end}",
                "text": sub.text.replace(chr(10), " ").strip(),
                "search_query": "",
                "description": "",
                "image_path": f"InputImage/{img}" if img else ""
            })

        return final

    # ───────────────── EXECUTION ─────────────────
    if not os.path.exists(SRT_FILE):
        sys.exit("❌ SRT file missing")

    if not os.path.exists(INPUT_IMAGE_JSON):
        sys.exit("❌ InputImage.json missing")

    subs = pysrt.open(SRT_FILE)

    with open(INPUT_IMAGE_JSON, "r", encoding="utf-8") as f:
        image_inventory = json.load(f)

    mapped = map_images_to_srt(subs, image_inventory, only_input_image=only_input_image)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(mapped, f, indent=2, ensure_ascii=False)

    return OUTPUT_FILE
