# ============================================================
# STAGE A + STAGE B
# WITH PRINTS + BASIC TRACKING
# (search queries + descriptions + image download)
# ============================================================

import json
import os
import sys
import time
from pathlib import Path
import pysrt
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from icrawler.builtin import BingImageCrawler

# Try importing ai_utils
try:
    from ai_utils import call_with_models, smart_text_call, get_ai_provider
except ImportError:
    try:
        from app.ai_utils import call_with_models, smart_text_call, get_ai_provider
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).resolve().parent))
        from ai_utils import call_with_models, smart_text_call, get_ai_provider

from prompts import get_search_query_prompt, get_visual_description_prompt, get_regenerate_descriptions_prompt

# ───────────────── ENV ─────────────────
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ───────────────── CONFIG ─────────────────
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
except Exception:
    CONFIG = {}

# Constants (fallback if config missing)
GEMINI_MAX_RETRIES = CONFIG.get("max_retries", 6)
GEMINI_BASE_DELAY = CONFIG.get("base_delay", 1.5)
IMAGE_EXTENSIONS = tuple(CONFIG.get("image_extensions", [".jpg", ".jpeg", ".png", ".webp"]))
TRACKING_FILE = CONFIG.get("tracking_file", "stage_progress.json")

TEXT_MODELS = CONFIG.get("text_models", ["gemini-2.5-flash"])
if isinstance(TEXT_MODELS, str): TEXT_MODELS = [TEXT_MODELS]

# ───────────────── UTILS ─────────────────
class PipelineStop(Exception):
    """Raised when pipeline wants to stop; used to allow one retry."""
    pass

def exit_pipeline(msg):
    # CHANGED: raise instead of sys.exit so we can retry once
    print(f"\n✖ PIPELINE STOPPED: {msg}")
    raise PipelineStop(msg)

def sleep_backoff(attempt):
    delay = GEMINI_BASE_DELAY * (2 ** (attempt - 1))
    print(f"⏳ Retry in {delay:.1f}s")
    time.sleep(delay)

def safe_folder_name(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

def find_first_image(folder: Path):
    if not folder.exists():
        return None
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return str(sorted(imgs)[0]) if imgs else None

def save_progress(step: str, index: int):
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump({"step": step, "last_index": index}, f, indent=2)

def load_progress():
    if Path(TRACKING_FILE).exists():
        with open(TRACKING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"step": None, "last_index": -1}

def gemini_text_call(prompt: str, task: str = "default") -> str:
    try:
        from ai_utils import smart_text_call
        return smart_text_call(prompt, task=task)
    except Exception as e:
        print(f"AI call failed for task {task}: {e}")
        raise e
def retry_gemini_and_parse(prompt: str, total: int, label: str, task: str = "default"):
    print(f"🔁 Retrying Gemini once due to missing {label}")
    raw_retry = gemini_text_call(prompt, task=task)
    return parse_indexed_lines(
        raw_retry,
        total,
        label,
        retry_prompt=True,
        prompt=prompt
    )

def parse_indexed_lines(
    raw: str,
    total: int,
    label: str,
    retry_prompt: bool = False,
    prompt = None,
    task: str = "default"
):
    result = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            idx = int(line.split("]", 1)[0][1:])
            val = line.split("]", 1)[1].strip()
            result[idx] = val
        except Exception:
            continue

    missing = [i for i in range(total) if i not in result]

    if missing:
        if not retry_prompt:
            # first failure → retry Gemini once
            return retry_gemini_and_parse(prompt, total, label, task=task)

        # already retried → hard stop
        exit_pipeline(f"Missing {label} for indexes: {missing}")

    return result



# ============================================================
# STEP 1 — SEARCH QUERIES (PROMPT UNCHANGED)
# ============================================================
def generate_search_queries(subs):
    print("▶ STEP 1: Generating search queries")

    subtitle_blocks = [
        f"""
[{i}]
Time: {sub.start} --> {sub.end}
Subtitle: {sub.text.replace(chr(10), ' ').strip()}
""".strip()
        for i, sub in enumerate(subs)
    ]

    subtitle_blocks_text = chr(10).join(subtitle_blocks)
    prompt = get_search_query_prompt(subtitle_blocks_text)

    raw = gemini_text_call(prompt, task="search_queries")
    queries = parse_indexed_lines(raw,len(subs),"search queries",retry_prompt=False,prompt=prompt, task="search_queries")
    print("✔ Search queries generated")
    return queries

# ============================================================
# STEP 2 — VISUAL DESCRIPTIONS (PROMPT UNCHANGED)
# ============================================================
def generate_descriptions(subs):
    print("▶ STEP 2: Generating visual descriptions")

    subtitle_blocks = [
        f"""
[{i}]
Time: {sub.start} --> {sub.end}
Subtitle: {sub.text.replace(chr(10), ' ').strip()}
""".strip()
        for i, sub in enumerate(subs)
    ]

    subtitle_blocks_text = chr(10).join(subtitle_blocks)
    prompt = get_visual_description_prompt(subtitle_blocks_text)

    raw = gemini_text_call(prompt, task="visual_descriptions")
    descriptions = parse_indexed_lines(raw, len(subs), "descriptions", retry_prompt=False, prompt=prompt, task="visual_descriptions")
    print("✔ Visual descriptions generated")
    return descriptions

def _normalize_desc_map(desc_map: dict, total: int) -> list:
    return [desc_map.get(i, "").strip() for i in range(total)]

def regenerate_missing_descriptions(subs, desc_list: list):
    missing = [i for i, d in enumerate(desc_list) if not d.strip()]
    if not missing:
        return desc_list

    print(f"⚠️ Missing descriptions at indexes: {missing}. Regenerating.")
    subtitle_blocks = [
        f"""
[{i}]
Time: {subs[i].start} --> {subs[i].end}
Subtitle: {subs[i].text.replace(chr(10), ' ').strip()}
""".strip()
        for i in missing
    ]

    subtitle_blocks_text = chr(10).join(subtitle_blocks)
    prompt = get_regenerate_descriptions_prompt(subtitle_blocks_text)

    raw = gemini_text_call(prompt, task="visual_descriptions")
    retry_map = parse_indexed_lines(raw, len(missing), "descriptions", retry_prompt=False, prompt=prompt, task="visual_descriptions")
    for local_idx, sub_idx in enumerate(missing):
        desc_list[sub_idx] = retry_map.get(local_idx, "").strip()
    return desc_list

# ============================================================
# STEP 3 — DOWNLOAD IMAGES (WITH PRINTS + TRACKING)
# ============================================================
def download_images(subs, queries, descriptions, output_file, skip_download=False):
    if skip_download or not CONFIG.get("features", {}).get("enable_image_download", True):
        print("▶ STEP 3: Skipping image download, generating description file only")
        stage2 = []
        for i in range(len(subs)):
            sub = subs[i]
            desc = (descriptions[i] or "").strip()
            if not desc:
                desc = f"Contextual illustrative news visual for: {sub.text.replace(chr(10), ' ').strip()}"
            stage2.append({
                "timeframe": f"{sub.start} --> {sub.end}",
                "text": sub.text.replace(chr(10), " ").strip(),
                "search_query": queries[i],
                "description": desc,
                "image_path": None
            })
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stage2, f, indent=2, ensure_ascii=False)
        return stage2

    print("▶ STEP 3: Downloading images")

    base = Path("downloaded_images")
    base.mkdir(exist_ok=True)

    progress = load_progress()
    start_index = (
        progress.get("last_index", -1) + 1
        if progress.get("step") == "download"
        else 0
    )

    stage2 = []

    for i in range(start_index, len(subs)):
        sub = subs[i]
        query = queries[i]

        folder = safe_folder_name(query)
        out_dir = base / folder
        out_dir.mkdir(exist_ok=True)

        print(f"[{i+1}/{len(subs)}] Downloading images for: {query}")

        crawler = BingImageCrawler(
            downloader_threads=1,
            storage={"root_dir": str(out_dir)}
        )

        try:
            crawler.crawl(keyword=query, max_num=1, min_size=(200, 200))
        except Exception as e:
            print(f"⚠ Download failed for '{query}': {e}")

        image_path = find_first_image(out_dir)

        stage2.append({
            "timeframe": f"{sub.start} --> {sub.end}",
            "text": sub.text.replace(chr(10), " ").strip(),
            "search_query": query,
            "description": descriptions[i],
            "image_path": image_path
        })

        save_progress("download", i)
        time.sleep(0.5)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stage2, f, indent=2, ensure_ascii=False)

    print("✔ Download_with_Description.json created")
    return stage2

# ============================================================
# RUN (ONLY TILL DOWNLOAD)
# ============================================================
if __name__ == "__main__":
    print("▶ Pipeline started")

    subs = pysrt.open("/Users/pragyan/Voicecline/app/runs/1769677523096/1769677523096.srt")
    if not subs:
        # This will be caught below and exit hard after retry
        exit_pipeline("SRT is empty")

    output_file = "/Users/pragyan/Voicecline/app/runs/1769677523096/1769677523096_Download_with_Description2.json"

    # CHANGED: only one retry if exit_pipeline occurs (e.g., missing indexes)
    for attempt in (1, 2):
        try:
            if attempt == 2:
                print("\n🔁 Retrying once after pipeline stop...\n")

            queries = generate_search_queries(subs)
            descriptions_map = generate_descriptions(subs)
            descriptions = _normalize_desc_map(descriptions_map, len(subs))
            descriptions = regenerate_missing_descriptions(subs, descriptions)
            download_images(subs, queries, descriptions, output_file)

            print("✔ Pipeline finished (download stage complete)")
            break

        except PipelineStop as e:
            if attempt == 2:
                # Final hard exit after single retry
                print(f"\n✖ PIPELINE STOPPED AFTER RETRY: {e}")
                sys.exit(1)
            # else: loop continues for the single retry
