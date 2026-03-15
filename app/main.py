# 1. VoiceCreate
# 2. Transcribe
# 3. ImageAnal
# 4. Input)image_json_map
# 5. DownloadGenImage
# 6. Imagen2.py
# 7. video_from_images

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pysrt

from DownloadImage import (
    download_images,
    generate_descriptions,
    generate_search_queries,
)
from ImageAnal import process_images
from Imagen2 import load_config, resolve_images
from Transcribe import transcribe_to_srt_limited
from input_image_json_map import run_mapping
from template import create_video
from voicecreate import generate_audio, generate_script
import pipeline_input as user_input


def get_unique_ms() -> int:
    return time.time_ns() // 1_000_000


def get_media_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def build_filtered_srt(input_srt: Path, output_srt: Path, skip_seconds: float) -> Path:
    subs = pysrt.open(str(input_srt))
    kept = pysrt.SubRipFile()
    for sub in subs:
        start_sec = sub.start.ordinal / 1000.0
        end_sec = sub.end.ordinal / 1000.0
        # Skip anything that starts before the prefix ends (avoid mapping inside prefix)
        if start_sec < skip_seconds or end_sec <= skip_seconds:
            continue
        kept.append(sub)
    kept.save(str(output_srt), encoding="utf-8")
    return output_srt


def build_empty_mapping_from_srt(srt_file, output_file):
    subs = pysrt.open(srt_file)
    empty_map = []
    for sub in subs:
        empty_map.append(
            {
                "timeframe": f"{sub.start} --> {sub.end}",
                "text": sub.text.replace(chr(10), " ").strip(),
                "search_query": "",
                "description": "",
                "image_path": "",
            }
        )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(empty_map, f, indent=2, ensure_ascii=False)
    return output_file


def build_loop_mapping_from_images(
    image_dir: Path,
    output_file: Path,
    total_duration: float,
    per_image_seconds: float,
    allowed_extensions=None,
):
    exts = allowed_extensions or [".jpg", ".jpeg", ".png", ".webp"]
    exts = {str(e).lower() for e in exts}
    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not images:
        raise RuntimeError(f"No input images found in {image_dir}")
    if per_image_seconds <= 0:
        per_image_seconds = 10.0
    if total_duration <= 0:
        total_duration = per_image_seconds

    timeline = []
    cursor = 0.0
    idx = 0
    while cursor < total_duration - 0.001:
        img = images[idx % len(images)]
        end = min(total_duration, cursor + per_image_seconds)
        timeline.append(
            {
                "timeframe": f"{sec_to_ts(cursor)} --> {sec_to_ts(end)}",
                "text": "",
                "search_query": "",
                "description": "",
                "image_path": str(img),
            }
        )
        cursor = end
        idx += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)
    return output_file


def get_prefix_skip_seconds(input_videos, input_videos_position, input_videos_audio):
    if input_videos_position == "prefix" and not input_videos_audio and input_videos:
        total = 0.0
        for item in input_videos:
            if isinstance(item, dict):
                video_path = item.get("path") or item.get("video_path")
            else:
                video_path = item
            if not video_path:
                continue
            dur = get_media_duration(Path(video_path))
            if dur > 0:
                total += dur
        return total
    return 0.0


def resolve_template_json(base_dir: Path, theme: str, orientation: str) -> Path:
    theme_norm = (theme or "").strip().lower()
    if theme_norm == "np":
        theme_norm = "newspick"

    if theme_norm == "failaan" and orientation == "horizontal":
        return base_dir / "template_failaan_horizontal.json"
    if theme_norm == "failaan":
        return base_dir / "template_failaan.json"
    if theme_norm == "newspick" and orientation == "horizontal":
        return base_dir / "template_newspick_horizontal.json"
    if theme_norm == "newspick":
        return base_dir / "template_newspick.json"
    if orientation == "horizontal":
        return base_dir / "template_horizontal.json"
    return base_dir / "template.json"


def sec_to_ts(sec: float) -> str:
    if sec < 0:
        sec = 0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ============================================================
# RUNTIME VARIABLES
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "config.json", "r", encoding="utf-8") as _cfg_f:
    _pipeline_config = json.load(_cfg_f)

features = _pipeline_config.get("features", {})
SKIP_DOWNLOAD = not features.get("enable_image_download", True)

CONTENT = str(getattr(user_input, "CONTENT", "")).strip()
if not CONTENT:
    sys.exit("pipeline_input.py: CONTENT is empty")

INPUT_THEME = str(getattr(user_input, "INPUT_THEME", "pg")).strip().lower()
AI_CONTENT = bool(getattr(user_input, "AI_CONTENT", True))
ENABLE_HEADING = bool(getattr(user_input, "ENABLE_HEADING", True))
HEADING_INPUT = getattr(user_input, "HEADING_INPUT", "")
SUBTITLES_ENABLED = bool(getattr(user_input, "SUBTITLES_ENABLED", True))
IMAGE_REVIEW_AI = bool(getattr(user_input, "IMAGE_REVIEW_AI", True))
IMAGE_LOOP_SECONDS = float(getattr(user_input, "IMAGE_LOOP_SECONDS", 10))

VOICE_NAMES = getattr(user_input, "VOICE_NAMES", ["Schedar"])
if isinstance(VOICE_NAMES, str):
    VOICE_NAMES = [VOICE_NAMES]
if not VOICE_NAMES:
    VOICE_NAMES = ["Schedar"]

VIDEO_ORIENTATION = str(getattr(user_input, "VIDEO_ORIENTATION", "vertical")).strip().lower()
if VIDEO_ORIENTATION not in ("vertical", "horizontal"):
    VIDEO_ORIENTATION = "vertical"

LANGUAGE = str(getattr(user_input, "LANGUAGE", _pipeline_config.get("language", "hindi"))).strip().lower()
if LANGUAGE not in ("hindi", "hinglish"):
    LANGUAGE = "hindi"

print("Initializing pipeline")

# Voiceover duration profile: "small", "medium", "large", "huge"
VOICEOVER_SIZE = str(getattr(user_input, "VOICEOVER_SIZE", "small")).strip().lower()
DURATION_PRESETS = {
    "small": (10, 30, "Under 30 seconds"),
    "medium": (30, 60, "30 seconds to 1 minute"),
    "large": (60, 120, "Up to 2 minutes"),
    "huge": (240, 300, "Around 5 minutes"),
}
if AI_CONTENT:
    if VOICEOVER_SIZE not in DURATION_PRESETS:
        raise ValueError(f"Invalid VOICEOVER_SIZE: {VOICEOVER_SIZE}")
    MIN_SEC, MAX_SEC, duration_label = DURATION_PRESETS[VOICEOVER_SIZE]
    print(f"Target duration: {duration_label} ({MIN_SEC}-{MAX_SEC} sec)")
else:
    MIN_SEC, MAX_SEC, duration_label = (0, 0, "Ignored when AI_CONTENT=False")
    print("Target duration: ignored because AI_CONTENT=False")

run_id = str(get_unique_ms())
print(f"Run ID generated: {run_id}")

# Directories
RUN_DIR = BASE_DIR / "runs" / run_id
FINAL_VIDEO_DIR = BASE_DIR / "final_videos"
INPUT_IMAGE_DIR = BASE_DIR / "InputImage"
CONFIG_JSON = BASE_DIR / "config.json"
TEMPLATE_THEME = INPUT_THEME.strip().lower()

print(f"Orientation: {VIDEO_ORIENTATION}")
print(f"Language: {LANGUAGE}")
print(f"Theme: {TEMPLATE_THEME}")
print(f"AI content refinement: {AI_CONTENT}")
print(f"Headings enabled: {ENABLE_HEADING}")
print(f"Subtitles enabled: {SUBTITLES_ENABLED}")
print(f"Image review AI: {IMAGE_REVIEW_AI}")

RUN_DIR.mkdir(parents=True, exist_ok=True)
FINAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Per-run files (inside RUN_DIR)
OUTPUT_WAV = RUN_DIR / f"{run_id}.wav"
SRT_FILE = RUN_DIR / f"{run_id}.srt"
FINAL_MAPPING_FILE = RUN_DIR / f"final_{run_id}.json"
STAGE2_FILE = RUN_DIR / f"{run_id}_Download_with_Description.json"
OUTPUT_WORDS = RUN_DIR / f"{run_id}_wordscaptions"

# Final video path (in final_videos)
FINAL_VIDEO = FINAL_VIDEO_DIR / f"final_video_{run_id}.mp4"

# Prefix/input video settings
raw_input_videos = getattr(user_input, "INPUT_VIDEOS", [])
input_videos = []
for item in raw_input_videos:
    if isinstance(item, dict):
        updated = dict(item)
        vp = updated.get("path") or updated.get("video_path")
        if vp:
            vp_path = Path(vp)
            if not vp_path.is_absolute():
                vp_path = BASE_DIR / vp_path
            if "path" in updated:
                updated["path"] = str(vp_path)
            else:
                updated["video_path"] = str(vp_path)
        input_videos.append(updated)
    else:
        vp_path = Path(str(item))
        if not vp_path.is_absolute():
            vp_path = BASE_DIR / vp_path
        input_videos.append(str(vp_path))

input_videos_position = str(getattr(user_input, "INPUT_VIDEOS_POSITION", "prefix")).strip().lower()
if input_videos_position not in ("prefix", "suffix"):
    input_videos_position = "prefix"

input_videos_audio = bool(getattr(user_input, "INPUT_VIDEOS_AUDIO", False))

default_audio_cfg = getattr(user_input, "DEFAULT_AUDIO", str(BASE_DIR / "BreakinNews-2.mp3"))
if default_audio_cfg:
    default_audio = Path(str(default_audio_cfg))
    if not default_audio.is_absolute():
        default_audio = BASE_DIR / default_audio
    default_audio = str(default_audio)
else:
    default_audio = None

# ============================================================
# PIPELINE
# ============================================================
print("Generating script and heading")
if AI_CONTENT:
    headings, script = generate_script(CONTENT, MIN_SEC, MAX_SEC, language=LANGUAGE)
else:
    headings = []
    script = CONTENT.strip()
    if not script:
        sys.exit("CONTENT is empty while AI_CONTENT is False")

if HEADING_INPUT:
    headings = HEADING_INPUT
if not ENABLE_HEADING:
    headings = ""

print(f"Headings: {headings if headings else 'disabled/empty'}")

time.sleep(5)
print("Generating audio")
generate_audio(script, OUTPUT_WAV, voice_names=VOICE_NAMES)

print("Transcribing audio to SRT")
transcribe_to_srt_limited(
    str(OUTPUT_WAV),
    str(SRT_FILE),
    str(OUTPUT_WORDS),
    language=LANGUAGE,
)

skip_seconds = get_prefix_skip_seconds(input_videos, input_videos_position, input_videos_audio)
SRT_FOR_MAPPING = SRT_FILE
if skip_seconds > 0:
    print(f"Prefix video detected (muted). Skipping mapping for first {skip_seconds:.2f}s")
    SRT_FOR_MAPPING = RUN_DIR / f"{run_id}_mapping.srt"
    build_filtered_srt(SRT_FILE, SRT_FOR_MAPPING, skip_seconds)

if not IMAGE_REVIEW_AI:
    print("IMAGE_REVIEW_AI is False. Building loop mapping from input images without AI review.")
    audio_duration = get_media_duration(OUTPUT_WAV)
    build_loop_mapping_from_images(
        image_dir=INPUT_IMAGE_DIR,
        output_file=FINAL_MAPPING_FILE,
        total_duration=audio_duration,
        per_image_seconds=IMAGE_LOOP_SECONDS,
        allowed_extensions=_pipeline_config.get("image_extensions", [".jpg", ".jpeg", ".png", ".webp"]),
    )
else:
    print("Processing input images")
    input_image_json = process_images(str(INPUT_IMAGE_DIR))

    with open(input_image_json, "r", encoding="utf-8") as f:
        input_image_inventory = json.load(f)

    has_input_images = bool(input_image_inventory)

    if has_input_images:
        if not features.get("enable_image_generation", True):
            print("Input images detected. Mapping subtitles to input images (use all).")
            run_mapping(
                SRT_FILE=SRT_FOR_MAPPING,
                INPUT_IMAGE_JSON=input_image_json,
                OUTPUT_FILE=FINAL_MAPPING_FILE,
                only_input_image=True,
            )
        else:
            print("Input images detected. Mapping each input image to best-fit subtitle.")
            run_mapping(
                SRT_FILE=SRT_FOR_MAPPING,
                INPUT_IMAGE_JSON=input_image_json,
                OUTPUT_FILE=FINAL_MAPPING_FILE,
                only_input_image=False,
            )
    elif not features.get("enable_image_generation", True):
        print("Image generation is disabled. Creating empty mapping.")
        build_empty_mapping_from_srt(SRT_FOR_MAPPING, FINAL_MAPPING_FILE)
    else:
        print("Image generation is enabled and no input images found. Creating empty mapping for generation.")
        build_empty_mapping_from_srt(SRT_FOR_MAPPING, FINAL_MAPPING_FILE)

    if features.get("enable_image_generation", True):
        print("Loading subtitles")
        subs = pysrt.open(str(SRT_FOR_MAPPING))
        if not subs:
            sys.exit("No subtitles found")

        print("Generating search queries")
        queries = generate_search_queries(subs)

        print("Generating descriptions")
        descriptions = generate_descriptions(subs)

        print("Downloading images")
        download_images(
            subs,
            queries,
            descriptions,
            STAGE2_FILE,
            skip_download=SKIP_DOWNLOAD,
        )

        # ============================================================
        # IMAGE RESOLUTION STAGE
        # ============================================================
        print("Loading config")
        config = load_config(
            config_path=str(CONFIG_JSON),
            stage2_file=str(STAGE2_FILE),
            final_mapping_file=str(FINAL_MAPPING_FILE),
        )
        # Use a per-run tracking file so previous runs don't skip generation
        config["tracking_file"] = str(RUN_DIR / "pipeline_progress.json")

        print("Loading stage2 data")
        with open(config["stage2_file"], "r", encoding="utf-8") as f:
            stage2_data = json.load(f)

        print("Resolving images")
        resolve_images(config, stage2_data)
    else:
        print("Image generation is disabled. Skipping download and generation stages.")

# ============================================================
# VIDEO CREATION STAGE
# ============================================================
create_video(
    timeline_json=str(FINAL_MAPPING_FILE),
    word_timeline_json=str(OUTPUT_WORDS),
    audio_file=str(OUTPUT_WAV),
    output_video=str(FINAL_VIDEO),
    heading_text=headings,
    voiceover_size=VOICEOVER_SIZE,
    input_videos=input_videos,
    input_videos_position=input_videos_position,
    input_videos_audio=input_videos_audio,
    default_audio=default_audio,
    theme=INPUT_THEME,
    video_orientation=VIDEO_ORIENTATION,
    language=LANGUAGE,
    heading_enabled=ENABLE_HEADING,
    subtitles_enabled=SUBTITLES_ENABLED,
)

if RUN_DIR.exists():
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    print(f"Deleted run directory: {RUN_DIR}")
