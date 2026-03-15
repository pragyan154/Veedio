import os
import time
import json
import subprocess
from pathlib import Path

import pysrt

from voicecreate import generate_script, generate_audio
from Transcribe import transcribe_to_srt_limited
from ImageAnal import process_images
from input_image_json_map import run_mapping
from DownloadImage import (
    generate_search_queries,
    generate_descriptions,
    download_images,
)
from Imagen2 import load_config, resolve_images
from template import create_video


def get_unique_ms():
    return time.time_ns() // 1_000_000


DURATION_PRESETS = {
    "small": (10, 30, "Under 30 seconds"),
    "medium": (30, 60, "30 seconds to 1 minute"),
    "large": (60, 120, "1 minute to 2 minutes"),
}


def get_media_duration(path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ],
        stdout=subprocess.PIPE,
        text=True
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def build_filtered_srt(input_srt: Path, output_srt: Path, skip_seconds: float) -> Path:
    subs = pysrt.open(str(input_srt))
    kept = pysrt.SubRipFile()
    for sub in subs:
        start_sec = sub.start.ordinal / 1000.0
        end_sec = sub.end.ordinal / 1000.0
        if start_sec < skip_seconds or end_sec <= skip_seconds:
            continue
        kept.append(sub)
    kept.save(str(output_srt), encoding="utf-8")
    return output_srt


def build_empty_mapping_from_srt(input_srt: Path, output_file: Path) -> Path:
    subs = pysrt.open(str(input_srt))
    empty_map = []
    for sub in subs:
        empty_map.append({
            "timeframe": f"{sub.start} --> {sub.end}",
            "text": sub.text.replace(chr(10), " ").strip(),
            "search_query": "",
            "description": "",
            "image_path": ""
        })
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(empty_map, f, indent=2, ensure_ascii=False)
    return output_file


def get_prefix_skip_seconds(input_videos, input_videos_position: str, input_videos_audio: bool) -> float:
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


class _ChDir:
    def __init__(self, path: Path):
        self.path = path
        self.prev = None

    def __enter__(self):
        self.prev = Path.cwd()
        os.chdir(self.path)

    def __exit__(self, exc_type, exc, tb):
        if self.prev:
            os.chdir(self.prev)


def _resolve_template_json(base_dir: Path, theme: str) -> Path:
    theme_norm = (theme or "").strip().lower()
    if theme_norm == "np":
        theme_norm = "newspick"
    if theme_norm == "failaan":
        candidate = base_dir / "template_failaan.json"
        if candidate.exists():
            return candidate
    if theme_norm == "newspick":
        candidate = base_dir / "template_newspick.json"
        if candidate.exists():
            return candidate
    return base_dir / "template.json"


def run_video_pipeline(
    content_text: str,
    voiceover_size: str = "medium",
    skip_download: bool = True,
    only_input_image: bool = False,
    input_image_dir=None,
    input_videos=None,
    input_videos_position: str = "prefix",
    input_videos_audio: bool = False,
    default_audio=None,
    theme: str = "pg",
):
    if not content_text or not isinstance(content_text, str):
        raise ValueError("content_text must be a non-empty string")

    if voiceover_size not in DURATION_PRESETS:
        raise ValueError(f"Invalid VOICEOVER_SIZE: {voiceover_size}")

    base_dir = Path(__file__).resolve().parent
    if input_image_dir is None:
        input_image_dir = base_dir / "InputImage"

    run_id = str(get_unique_ms())

    run_dir = base_dir / "runs" / run_id
    final_video_dir = base_dir / "final_videos"
    input_image_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    final_video_dir.mkdir(parents=True, exist_ok=True)

    output_wav = run_dir / f"{run_id}.wav"
    srt_file = run_dir / f"{run_id}.srt"
    final_mapping_file = run_dir / f"final_{run_id}.json"
    stage2_file = run_dir / f"{run_id}_Download_with_Description.json"
    output_words = run_dir / f"{run_id}_wordscaptions"
    final_video = final_video_dir / f"final_video_{run_id}.mp4"

    min_sec, max_sec, duration_label = DURATION_PRESETS[voiceover_size]
    print(f"Target duration: {duration_label} ({min_sec}-{max_sec} sec)")

    print("Generating script and heading")
    headings, script = generate_script(content_text, min_sec, max_sec)

    print("Generating audio")
    generate_audio(script, output_wav)

    print("Transcribing audio to SRT")
    transcribe_to_srt_limited(str(output_wav), str(srt_file), str(output_words))

    skip_seconds = get_prefix_skip_seconds(input_videos, input_videos_position, input_videos_audio)
    srt_for_mapping = srt_file
    if skip_seconds > 0:
        print(f"Prefix video detected (muted). Skipping mapping for first {skip_seconds:.2f}s")
        srt_for_mapping = run_dir / f"{run_id}_mapping.srt"
        build_filtered_srt(srt_file, srt_for_mapping, skip_seconds)

    print("Processing input images")
    input_image_json = process_images(str(input_image_dir))

    with open(input_image_json, "r", encoding="utf-8") as f:
        input_image_inventory = json.load(f)

    has_input_images = bool(input_image_inventory)

    if has_input_images:
        if only_input_image:
            print("Input images detected. Mapping subtitles to input images (use all).")
            run_mapping(
                SRT_FILE=srt_for_mapping,
                INPUT_IMAGE_JSON=input_image_json,
                OUTPUT_FILE=final_mapping_file,
                only_input_image=True
            )
        else:
            print("Input images detected. Mapping each input image to best-fit subtitle.")
            run_mapping(
                SRT_FILE=srt_for_mapping,
                INPUT_IMAGE_JSON=input_image_json,
                OUTPUT_FILE=final_mapping_file,
                only_input_image=False
            )
    elif only_input_image:
        print("ONLY_INPUT_IMAGE is True but no input images found. Creating empty mapping.")
        build_empty_mapping_from_srt(srt_for_mapping, final_mapping_file)
    else:
        print("ONLY_INPUT_IMAGE is False and no input images found. Creating empty mapping for generation.")
        build_empty_mapping_from_srt(srt_for_mapping, final_mapping_file)

    if not only_input_image:
        print("Loading subtitles")
        subs = pysrt.open(str(srt_for_mapping))
        if not subs:
            raise RuntimeError("No subtitles found")

        print("Generating search queries")
        queries = generate_search_queries(subs)

        print("Generating descriptions")
        descriptions = generate_descriptions(subs)

        print("Downloading images")
        download_images(
            subs,
            queries,
            descriptions,
            stage2_file,
            skip_download=skip_download
        )

        print("Loading config")
        config = load_config(
            config_path=str(base_dir / "config.json"),
            stage2_file=str(stage2_file),
            final_mapping_file=str(final_mapping_file),
        )
        config["tracking_file"] = str(run_dir / "pipeline_progress.json")

        print("Loading stage2 data")
        with open(config["stage2_file"], "r", encoding="utf-8") as f:
            stage2_data = json.load(f)

        print("Resolving images")
        resolve_images(config, stage2_data)
    else:
        print("ONLY_INPUT_IMAGE is True. Skipping download and generation stages.")

    if default_audio is None:
        default_audio = base_dir / "BreakinNews-2.mp3"

    print("Creating final video")
    template_json = _resolve_template_json(base_dir, theme)
    with _ChDir(base_dir):
        create_video(
            timeline_json=str(final_mapping_file),
            word_timeline_json=str(output_words),
            audio_file=str(output_wav),
            output_video=str(final_video),
            template_json=str(template_json),
            heading_text=headings,
            voiceover_size=voiceover_size,
            input_videos=input_videos,
            input_videos_position=input_videos_position,
            input_videos_audio=input_videos_audio,
            default_audio=str(default_audio) if default_audio else None
        )

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "output_video": str(final_video),
        "heading": headings,
    }
