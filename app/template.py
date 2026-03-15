import cv2
import json
import numpy as np
import subprocess
import os
import re
import random
from PIL import Image, ImageDraw, ImageFont
import textwrap

from pathlib import Path

# ================= FILES =================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Default paths (used only when running this file directly)
TIMELINE_JSON = BASE_DIR / "/Users/pragyan/Voicecline/app/runs/1769771114024/final_1769771114024.json"                 # image + timeframe mapping
WORD_TIMELINE_JSON = BASE_DIR / "/Users/pragyan/Voicecline/app/runs/1769771114024/1769771114024_wordscaptions"   # word-level timestamps
TEMPLATE_JSON = BASE_DIR / "template.json"
AUDIO_FILE = BASE_DIR / "/Users/pragyan/Voicecline/app/runs/1769771114024/1769771114024.wav"
OUTPUT_VIDEO = BASE_DIR / "/Users/pragyan/Voicecline/app/final_videos/17697711140241769771114024.mp4"
TMP_VIDEO = BASE_DIR / "_tmp.mp4"
TMP_MAIN_VIDEO = BASE_DIR / "_tmp_main.mp4"
TMP_MAIN_WITH_AUDIO = BASE_DIR / "_tmp_main_with_audio.mp4"
TMP_PREFIX_SUFFIX = BASE_DIR / "_tmp_prefix_suffix.mp4"
TMP_CONCAT_VIDEO = BASE_DIR / "_tmp_concat_video.mp4"
TMP_SUFFIX_AUDIO = BASE_DIR / "_tmp_suffix_audio.m4a"
TMP_FULL_AUDIO = BASE_DIR / "_tmp_full_audio.aac"
TMP_FALLBACK_IMAGE = BASE_DIR / "_tmp_fallback.png"

# ================= FONT =================
def _resolve_font_path():
    candidates = [
        PROJECT_ROOT / "Noto_Sans_Devanagari-2" / "static" / "NotoSansDevanagari_Condensed-Bold.ttf",
        PROJECT_ROOT / "card-new" / "Font" / "Noto_Sans_Devanagari" / "static" / "NotoSansDevanagari-Regular.ttf",
        BASE_DIR / "Noto_Sans_Devanagari-2" / "static" / "NotoSansDevanagari_Condensed-Bold.ttf",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

FONT_PATH = _resolve_font_path()
SUBTITLE_FONT_PATH = FONT_PATH
HEADING_FONT_PATH = FONT_PATH
FONT_SIZE = 62
SUB_FONT_SIZE = FONT_SIZE
SUB_STROKE_WIDTH = 0

SUB_BG = (0, 0, 0, 180)
SUB_TEXT = (200, 200, 200, 255)
SUB_ACTIVE = (255, 220, 80, 255)

SUB_PADDING = 18
SUB_RADIUS = 25

WINDOW_SIZE = 8  # fixed window size

# ================= HELPERS =================
def ts_to_sec(ts):
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def parse_timeframe(tf):
    s, e = tf.split("-->")
    return ts_to_sec(s.strip()), ts_to_sec(e.strip())

def sec_to_ts(sec):
    if sec < 0:
        sec = 0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_sequential_entries(entries, max_duration=None, offset=0.0):
    out = []
    cursor = 0.0
    for entry in entries:
        start, end = parse_timeframe(entry["timeframe"])
        dur = max(0.0, end - start)
        if dur <= 0:
            continue
        if max_duration is not None:
            remaining = max_duration - cursor
            if remaining <= 0:
                break
            if dur > remaining:
                dur = remaining
        new_entry = dict(entry)
        new_start = offset + cursor
        new_end = offset + cursor + dur
        new_entry["timeframe"] = f"{sec_to_ts(new_start)} --> {sec_to_ts(new_end)}"
        out.append(new_entry)
        cursor += dur
    return out, cursor

def fit_contain(img, bw, bh):
    h, w = img.shape[:2]
    ar_i = w / h
    ar_b = bw / bh
    if ar_i > ar_b:
        nw = bw
        nh = int(bw / ar_i)
    else:
        nh = bh
        nw = int(bh * ar_i)
    return cv2.resize(img, (nw, nh))

def blur_fill_from_media(img, bw, bh):
    bg = cv2.resize(img, (bw, bh))
    return cv2.GaussianBlur(bg, (51, 51), 0)


def _center_crop_resize(img, out_w, out_h):
    h, w = img.shape[:2]
    if w == out_w and h == out_h:
        return img
    x = max(0, (w - out_w) // 2)
    y = max(0, (h - out_h) // 2)
    cropped = img[y:y + out_h, x:x + out_w]
    if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
        cropped = cv2.resize(cropped, (out_w, out_h))
    return cropped


def _zoom_frame(img, progress):
    h, w = img.shape[:2]
    scale = 1.12 - (0.12 * max(0.0, min(1.0, progress)))
    zw = int(round(w * scale))
    zh = int(round(h * scale))
    zoomed = cv2.resize(img, (zw, zh))
    return _center_crop_resize(zoomed, w, h)


def apply_transition(prev_frame, curr_frame, transition_name, progress):
    if prev_frame is None or curr_frame is None:
        return curr_frame
    if transition_name in ("", "none", "cut", None):
        return curr_frame

    p = max(0.0, min(1.0, float(progress)))
    h, w = curr_frame.shape[:2]

    if transition_name == "fade":
        return cv2.addWeighted(prev_frame, 1.0 - p, curr_frame, p, 0.0)

    if transition_name == "wipe_left":
        out = prev_frame.copy()
        edge = int(w * p)
        if edge > 0:
            out[:, :edge] = curr_frame[:, :edge]
        return out

    if transition_name == "wipe_right":
        out = prev_frame.copy()
        edge = int(w * p)
        if edge > 0:
            out[:, w - edge:] = curr_frame[:, w - edge:]
        return out

    if transition_name == "slide_left":
        out = prev_frame.copy()
        offset = int((1.0 - p) * w)
        if offset <= 0:
            return curr_frame
        out[:, :offset] = prev_frame[:, w - offset:]
        out[:, offset:] = curr_frame[:, :w - offset]
        return out

    if transition_name == "zoom":
        zoomed = _zoom_frame(curr_frame, p)
        return cv2.addWeighted(prev_frame, 1.0 - p, zoomed, p, 0.0)

    return curr_frame

def is_video_path(path):
    return str(path).lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))

def get_media_duration(path):
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

def render_word_window(word_list, active_idx, max_width):
    try:
        font = ImageFont.truetype(str(SUBTITLE_FONT_PATH), SUB_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    padding_x = SUB_PADDING
    padding_y = SUB_PADDING
    line_gap = 10

    tmp = Image.new("RGBA", (1, 1))
    d_tmp = ImageDraw.Draw(tmp)

    lines = [[]]
    for i, w in enumerate(word_list):
        test = " ".join(lines[-1] + [w])
        if d_tmp.textlength(test, font=font) <= (max_width - 2 * padding_x):
            lines[-1].append(w)
        else:
            lines.append([w])

    lines = lines[:2]  # max 2 lines

    line_heights = []
    for line in lines:
        bbox = d_tmp.textbbox((0, 0), " ".join(line), font=font)
        line_heights.append(bbox[3] - bbox[1])

    img_h = sum(line_heights) + padding_y * 2 + line_gap * (len(lines) - 1)
    img = Image.new("RGBA", (max_width, img_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.rounded_rectangle(
        (0, 0, max_width, img_h),
        radius=SUB_RADIUS,
        fill=SUB_BG
    )

    y = padding_y
    word_counter = 0
    for li, line in enumerate(lines):
        widths = [d.textlength(w, font=font) for w in line]
        space = d.textlength(" ", font=font)
        total_w = sum(widths) + space * (len(line) - 1)
        x = (max_width - total_w) // 2

        for wi, w in enumerate(line):
            color = SUB_ACTIVE if word_counter == active_idx else SUB_TEXT
            d.text(
                (x, y),
                w,
                font=font,
                fill=color,
                stroke_width=SUB_STROKE_WIDTH,
                stroke_fill=color,
            )
            x += widths[wi] + space
            word_counter += 1

        y += line_heights[li] + line_gap

    return np.array(img)


def render_heading(text, max_width, heading_cfg):
    # allow overriding font path in heading config (optional)
    font_path = heading_cfg.get("font_path")
    if font_path:
        font_path = Path(font_path)
        if not font_path.is_absolute():
            candidate = PROJECT_ROOT / font_path
            if candidate.exists():
                font_path = candidate
    else:
        font_path = HEADING_FONT_PATH
    if not text:
        return None

    font_size = heading_cfg.get("font_size", 50)
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except OSError:
        font = ImageFont.load_default()
    padding_x = heading_cfg.get("padding_x", 10)
    padding_y = heading_cfg.get("padding_y", 10)

    # temp draw context
    tmp = Image.new("RGBA", (1, 1))
    d_tmp = ImageDraw.Draw(tmp)

    # --- wrap text by pixel width ---
    lines = []
    current = ""
    for word in text.split():
        test = current + (" " if current else "") + word
        if d_tmp.textlength(test, font=font) <= (max_width - 2 * padding_x):
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    # --- measure text ---
    line_heights = []
    for line in lines:
        bbox = d_tmp.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])

    total_h = sum(line_heights)
    img_h = total_h + 2 * padding_y

    img = Image.new("RGBA", (max_width, img_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    bg_color = tuple(heading_cfg.get("bg_color", [255, 0, 0, 255]))
    text_color = tuple(heading_cfg.get("text_color", [255, 255, 255, 255]))

    # background
    d.rectangle((0, 0, max_width, img_h), fill=bg_color)

    # --- centered multiline draw ---
    y = padding_y
    for i, line in enumerate(lines):
        w = d.textlength(line, font=font)
        x = (max_width - w) // 2
        d.text((x, y), line, font=font, fill=text_color)
        y += line_heights[i]

    return np.array(img)


def normalize_headings(heading_text):
    if isinstance(heading_text, (list, tuple)):
        items = []
        for h in heading_text:
            s = str(h).strip()
            if s:
                items.append(s)
        return items
    if not heading_text:
        return []
    text = str(heading_text).strip()
    if "\n" in text:
        items = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            line = re.sub(r'^(?:\d+[.)]|\-|\*)\s*', "", line).strip()
            if line:
                items.append(line)
        return items if items else [text]
    return [text]

# ================= AUDIO HELPERS =================
def apply_default_audio_after_voiceover(output_video, voiceover_audio, default_audio):
    if not default_audio:
        return
    if not Path(voiceover_audio).exists():
        print("⚠️ voiceover audio missing; skipping default audio append.")
        return

    video_dur = get_media_duration(output_video)
    voice_dur = get_media_duration(voiceover_audio)
    if video_dur <= 0 or voice_dur <= 0:
        return

    if video_dur <= voice_dur + 0.01:
        # Ensure only voiceover is present, trimmed to video duration
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(output_video),
            "-i", str(voiceover_audio),
            "-filter_complex", f"[1:a]atrim=0:{video_dur}[a]",
            "-map", "0:v:0",
            "-map", "[a]",
            "-c:v", "libx264",
            "-c:a", "aac",
            str(TMP_PREFIX_SUFFIX)
        ])
        os.replace(TMP_PREFIX_SUFFIX, output_video)
        return

    remain = max(0.0, video_dur - voice_dur)
    if remain <= 0.01:
        return

    def _run_ffmpeg(args):
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            print(f"⚠️ ffmpeg failed: {' '.join(args)}")
            if proc.stderr:
                print(proc.stderr.strip())
            return False
        return True

    # Build looping default audio for the remaining duration
    if not _run_ffmpeg([
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", str(default_audio),
        "-t", str(remain),
        "-c:a", "aac",
        str(TMP_SUFFIX_AUDIO)
    ]):
        return
    if not os.path.exists(TMP_SUFFIX_AUDIO) or os.path.getsize(TMP_SUFFIX_AUDIO) == 0:
        print("⚠️ Failed to build default audio segment; skipping.")
        return

    # Concatenate voiceover + default audio
    if not _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(voiceover_audio),
        "-i", str(TMP_SUFFIX_AUDIO),
        "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[a]",
        "-map", "[a]",
        "-c:a", "aac",
        str(TMP_FULL_AUDIO)
    ]):
        return
    if not os.path.exists(TMP_FULL_AUDIO) or os.path.getsize(TMP_FULL_AUDIO) == 0:
        print("⚠️ Failed to concatenate voiceover + default audio; skipping.")
        return

    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(output_video),
        "-i", str(TMP_FULL_AUDIO),
        "-filter_complex", f"[1:a]apad,atrim=0:{video_dur}[a]",
        "-map", "0:v:0",
        "-map", "[a]",
        "-c:v", "libx264",
        "-c:a", "aac",
        str(TMP_PREFIX_SUFFIX)
    ])
    if os.path.exists(TMP_PREFIX_SUFFIX):
        os.replace(TMP_PREFIX_SUFFIX, output_video)
    else:
        print("⚠️ Failed to apply default audio to final video.")

# ================= LOGO HELPERS =================
def load_logo(logo_cfg, project_root):
    """Load and scale logo image. Returns (logo_bgra, x, y) or None."""
    if not logo_cfg or not logo_cfg.get("enabled", False):
        return None

    logo_path = logo_cfg.get("path")
    if not logo_path:
        # look for logo.png in project root
        candidates = [
            Path(project_root) / "logo.png",
            Path(project_root) / "logo.jpg",
            Path(project_root) / "app" / "logo.png",
        ]
        for c in candidates:
            if c.exists():
                logo_path = str(c)
                break
    if not logo_path or not Path(logo_path).exists():
        return None

    img = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Ensure BGRA
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    max_w = logo_cfg.get("max_width", 120)
    max_h = logo_cfg.get("max_height", 60)
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    return img

def overlay_logo(bg, logo_bgra, logo_cfg, canvas_width, canvas_height):
    """Composite logo onto bg at the configured position."""
    if logo_bgra is None:
        return

    lh, lw = logo_bgra.shape[:2]
    x_margin = logo_cfg.get("x_margin", 20)
    y_margin = logo_cfg.get("y_margin", 20)
    position = logo_cfg.get("position", "top-right")

    if position == "top-left":
        lx, ly = x_margin, y_margin
    elif position == "top-right":
        lx, ly = canvas_width - lw - x_margin, y_margin
    elif position == "bottom-left":
        lx, ly = x_margin, canvas_height - lh - y_margin
    elif position == "bottom-right":
        lx, ly = canvas_width - lw - x_margin, canvas_height - lh - y_margin
    else:
        lx, ly = canvas_width - lw - x_margin, y_margin

    # Clamp
    lx = max(0, min(lx, canvas_width - lw))
    ly = max(0, min(ly, canvas_height - lh))

    alpha = logo_bgra[:, :, 3] / 255.0
    for c in range(3):
        bg[ly:ly+lh, lx:lx+lw, c] = (
            alpha * logo_bgra[:, :, c] +
            (1 - alpha) * bg[ly:ly+lh, lx:lx+lw, c]
        )

# ================= RENDER CORE =================
def render_timeline_video(
    timeline,
    words,
    output_path,
    template,
    heading_text="",
    render_subtitles=True
):
    WIDTH = template["canvas"]["width"]
    HEIGHT = template["canvas"]["height"]
    FPS = template["canvas"]["fps"]

    BG_CFG = template.get("background", {})
    MEDIA_BOX = template["main_media"]
    SUB_CFG = template["subtitle"]
    HEADING_CFG = template.get("heading", {})

    LOGO_CFG = template.get("logo", {})
    TRANSITION_CFG = template.get("transitions", {})

    if render_subtitles and not words:
        raise ValueError("Word timeline empty")

    transition_enabled = bool(TRANSITION_CFG.get("enabled", True))
    transition_duration_sec = float(TRANSITION_CFG.get("duration_sec", 0.35))
    transition_choices = TRANSITION_CFG.get(
        "choices",
        ["", "cut", "fade", "wipe_left", "wipe_right", "zoom", "slide_left"],
    )
    if isinstance(transition_choices, str):
        transition_choices = [transition_choices]
    if not transition_choices:
        transition_choices = ["cut"]
    entry_transitions = ["cut"]
    if transition_enabled:
        for _ in range(max(0, len(timeline) - 1)):
            entry_transitions.append(random.choice(transition_choices))
    else:
        entry_transitions = ["cut"] * max(1, len(timeline))

    headings = normalize_headings(heading_text)
    if not headings:
        headings = normalize_headings(HEADING_CFG.get("text", ""))

    # Load logo
    logo_img = load_logo(LOGO_CFG, PROJECT_ROOT)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT)
    )

    # Background init
    bg_img = None
    bg_cap = None
    bg_src = BG_CFG.get("source_from")

    if bg_src and os.path.exists(bg_src):
        if bg_src.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            bg_cap = cv2.VideoCapture(bg_src)
        else:
            bg_img = cv2.imread(bg_src)

    # Media box
    bw = int(WIDTH * MEDIA_BOX["w"])
    bh = int(HEIGHT * MEDIA_BOX["h"])
    bx = int(WIDTH * MEDIA_BOX["x"])
    by = int(HEIGHT * MEDIA_BOX["y"])

    # Pre-render headings
    heading_layers = [render_heading(h, WIDTH, HEADING_CFG) for h in headings]
    heading_layers = [h for h in heading_layers if h is not None]
    heading_count = len(heading_layers)

    total_duration = 0.0
    for entry in timeline:
        _, end = parse_timeframe(entry["timeframe"])
        if end > total_duration:
            total_duration = end
    segment_len = (total_duration / heading_count) if heading_count > 0 else 0.0
    intro_seconds = float(HEADING_CFG.get("intro_seconds", 0.4))
    exit_seconds = float(HEADING_CFG.get("exit_seconds", 0.4))

    # Subtitle state
    word_idx = 0
    window_start_idx = 0
    total_words = len(words) if words else 0
    prev_entry_box = None

    for entry_idx, entry in enumerate(timeline):
        start, end = parse_timeframe(entry["timeframe"])
        start_f = int(start * FPS)
        end_f = int(end * FPS)

        media_path = (
            entry.get("image_or_video_path")
            or entry.get("video_path")
            or entry.get("image_path")
        )
        if not media_path:
            continue

        media_path = Path(media_path)
        if not media_path.exists():
            continue

        is_video = is_video_path(media_path)
        cap = None
        main_img = None
        last_frame = None
        if is_video:
            cap = cv2.VideoCapture(str(media_path))
        else:
            main_img = cv2.imread(str(media_path))
            if main_img is None:
                continue
        transition_name = entry_transitions[entry_idx] if entry_idx < len(entry_transitions) else "cut"
        transition_frames = max(1, int(round(transition_duration_sec * FPS)))
        last_entry_box = None

        for frame_idx in range(start_f, end_f):
            t = frame_idx / FPS

            if render_subtitles:
                while word_idx + 1 < total_words and t > words[word_idx]["end"]:
                    word_idx += 1

                # ---- WINDOW SHIFT (ONLY WHEN WINDOW ENDS) ----
                if word_idx >= window_start_idx + WINDOW_SIZE:
                    window_start_idx = word_idx

                window_end = min(total_words, window_start_idx + WINDOW_SIZE)
                window_words = [w["word"] for w in words[window_start_idx:window_end]]
                active_local_idx = word_idx - window_start_idx

            # ---- BACKGROUND ----
            if bg_cap:
                ret, bg_frame = bg_cap.read()
                if not ret and BG_CFG.get("loop", False):
                    bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, bg_frame = bg_cap.read()
                bg = cv2.resize(bg_frame, (WIDTH, HEIGHT)) if ret else np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            elif bg_img is not None:
                bg = cv2.resize(bg_img, (WIDTH, HEIGHT))
            else:
                bg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            if BG_CFG.get("blur", False):
                bg = cv2.GaussianBlur(bg, (51, 51), 0)

            # ---- MAIN MEDIA ----
            if is_video:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if ret:
                    main_img = frame
                    last_frame = frame
                elif last_frame is not None:
                    main_img = last_frame
                else:
                    main_img = np.zeros((bh, bw, 3), dtype=np.uint8)

            frame_box = blur_fill_from_media(main_img, bw, bh)
            media = fit_contain(main_img, bw, bh)
            mx = (bw - media.shape[1]) // 2
            my = (bh - media.shape[0]) // 2
            frame_box[my:my+media.shape[0], mx:mx+media.shape[1]] = media
            if transition_enabled and entry_idx > 0 and prev_entry_box is not None:
                local_idx = frame_idx - start_f
                if local_idx < transition_frames:
                    progress = (local_idx + 1) / float(transition_frames)
                    frame_box = apply_transition(prev_entry_box, frame_box, transition_name, progress)
            bg[by:by+bh, bx:bx+bw] = frame_box
            last_entry_box = frame_box

            # ---- HEADING ----
            if heading_layers:
                idx = 0
                if heading_count > 1 and segment_len > 0:
                    idx = int(t / segment_len)
                    if idx >= heading_count:
                        idx = heading_count - 1
                heading_layer = heading_layers[idx]

                hh = heading_layer.shape[0]
                hy = HEADING_CFG.get("y_margin_top", 50)

                alpha_mult = 1.0
                y_offset = 0
                if heading_count > 0 and segment_len > 0:
                    seg_start = idx * segment_len
                    local_t = t - seg_start
                    max_fx = segment_len * 0.45
                    intro = min(intro_seconds, max_fx)
                    outro = min(exit_seconds, max_fx)
                    if intro > 0 and local_t < intro:
                        p = max(0.0, min(1.0, local_t / intro))
                        alpha_mult = p
                        y_offset = int((1 - p) * 20)
                    elif outro > 0 and local_t > segment_len - outro:
                        p = max(0.0, min(1.0, (segment_len - local_t) / outro))
                        alpha_mult = p
                        y_offset = int((1 - p) * 20)

                hy = hy + y_offset
                if hy < 0:
                    hy = 0
                if hy + hh > HEIGHT:
                    hh = max(0, HEIGHT - hy)

                if hh > 0:
                    # RGBA → BGR
                    heading_rgb = heading_layer[:hh, :, :3][:, :, ::-1]
                    h_alpha = (heading_layer[:hh, :, 3] / 255.0) * alpha_mult

                    for c in range(3):
                        bg[hy:hy+hh, :, c] = (
                            h_alpha * heading_rgb[:, :, c] +
                            (1 - h_alpha) * bg[hy:hy+hh, :, c]
                        )

            # ---- SUBTITLE ----
            if render_subtitles:
                sub_img = render_word_window(window_words, active_local_idx, WIDTH)
                sh = sub_img.shape[0]
                sy = HEIGHT - sh - SUB_CFG["y_margin_bottom"]
                if sy < 0:
                    sy = 0
                if sy + sh > HEIGHT:
                    sh = max(0, HEIGHT - sy)
                    sub_img = sub_img[:sh, :, :]
                if sh <= 0:
                    writer.write(bg)
                    continue
                alpha = sub_img[:, :, 3] / 255.0

                for c in range(3):
                    bg[sy:sy+sh, :, c] = (
                        alpha * sub_img[:, :, c] +
                        (1 - alpha) * bg[sy:sy+sh, :, c]
                    )

            # ---- LOGO OVERLAY ----
            if logo_img is not None:
                overlay_logo(bg, logo_img, LOGO_CFG, WIDTH, HEIGHT)

            writer.write(bg)

        if cap:
            cap.release()
        if last_entry_box is not None:
            prev_entry_box = last_entry_box

    if bg_cap:
        bg_cap.release()

    writer.release()

# ================= FUNCTIONAL ENTRYPOINT =================
def create_video(
    timeline_json,
    word_timeline_json,
    audio_file,
    output_video,
    template_json=None,
    heading_text="",
    input_videos=None,
    input_videos_position="prefix",
    input_videos_audio=False,
    default_audio=None,
    voiceover_size=None,
    theme="pg",
    video_orientation="vertical",
    language="hindi",
    heading_enabled=True,
    subtitles_enabled=True,
):
    # Normalize to Path objects
    timeline_json = Path(timeline_json)
    word_timeline_json = Path(word_timeline_json)
    audio_file = Path(audio_file)
    output_video = Path(output_video)
    if template_json:
        template_json = Path(template_json)
    else:
        base_dir = Path(__file__).resolve().parent
        theme_norm = (theme or "").strip().lower()
        orientation_norm = (video_orientation or "").strip().lower()
        if theme_norm == "np":
            theme_norm = "newspick"
        if orientation_norm not in ("vertical", "horizontal"):
            orientation_norm = "vertical"

        if theme_norm == "failaan" and orientation_norm == "horizontal":
            template_json = base_dir / "template_failaan_horizontal.json"
        elif theme_norm == "failaan":
            template_json = base_dir / "template_failaan.json"
        elif theme_norm == "newspick" and orientation_norm == "horizontal":
            template_json = base_dir / "template_newspick_horizontal.json"
        elif theme_norm == "newspick":
            template_json = base_dir / "template_newspick.json"
        elif orientation_norm == "horizontal":
            template_json = base_dir / "template_horizontal.json"
        else:
            template_json = base_dir / "template.json"
    if default_audio:
        default_audio = Path(default_audio)
        if not default_audio.exists():
            default_audio = None

    # ================= LOAD DATA =================
    with open(template_json, "r") as f:
        template = json.load(f)

    global SUBTITLE_FONT_PATH, HEADING_FONT_PATH, SUB_FONT_SIZE, SUB_STROKE_WIDTH
    subtitle_font_path = template.get("subtitle", {}).get("font_path")
    heading_font_path = template.get("heading", {}).get("font_path")

    if subtitle_font_path:
        subtitle_font_path = Path(subtitle_font_path)
        if not subtitle_font_path.is_absolute():
            subtitle_font_path = PROJECT_ROOT / subtitle_font_path
    if heading_font_path:
        heading_font_path = Path(heading_font_path)
        if not heading_font_path.is_absolute():
            heading_font_path = PROJECT_ROOT / heading_font_path

    if not subtitle_font_path or not subtitle_font_path.exists():
        if (language or "").strip().lower() == "hinglish":
            subtitle_candidates = [
                BASE_DIR / "fonts" / "NotoSans-VF.ttf",
                Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
                Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            ]
        else:
            subtitle_candidates = [
                BASE_DIR / "fonts" / "NotoSansDevanagari-VF.ttf",
                BASE_DIR / "fonts" / "NotoSans-VF.ttf",
                Path("/System/Library/Fonts/Supplemental/NotoSansDevanagari.ttc"),
                Path("/System/Library/Fonts/Supplemental/KohinoorDevanagari.ttc"),
            ]
        subtitle_font_path = next((p for p in subtitle_candidates if p.exists()), FONT_PATH)

    if not heading_font_path or not heading_font_path.exists():
        heading_font_path = subtitle_font_path

    SUBTITLE_FONT_PATH = subtitle_font_path
    HEADING_FONT_PATH = heading_font_path
    subtitle_cfg = template.get("subtitle", {})
    SUB_FONT_SIZE = int(subtitle_cfg.get("font_size", FONT_SIZE))
    SUB_STROKE_WIDTH = 1 if subtitle_cfg.get("bold", False) else 0

    with open(timeline_json, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    with open(word_timeline_json, "r", encoding="utf-8") as f:
        words = json.load(f)

    # If heading is disabled, force empty heading.
    if not heading_enabled:
        heading_text = ""
    # If heading_text is not provided, try to get it from template
    if heading_enabled and not heading_text:
        heading_text = template.get("heading", {}).get("text", "")

    # Respect per-item video_audio overrides
    effective_input_videos_audio = bool(input_videos_audio)
    if input_videos:
        for item in input_videos:
            if isinstance(item, dict) and item.get("video_audio") is True:
                effective_input_videos_audio = True
                break

    def _get_entry_media_path(entry):
        return (
            entry.get("image_or_video_path")
            or entry.get("video_path")
            or entry.get("image_path")
        )

    # Filter out timeline entries that point to missing media
    timeline_valid = []
    for entry in timeline:
        media_path = _get_entry_media_path(entry)
        if not media_path:
            continue
        if Path(media_path).exists():
            timeline_valid.append(entry)

    # If nothing valid to render, create a fallback single-frame timeline
    if not timeline_valid:
        audio_dur = get_media_duration(audio_file)
        if audio_dur <= 0:
            audio_dur = 1.0
        width = int(template["canvas"]["width"])
        height = int(template["canvas"]["height"])
        fallback_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(TMP_FALLBACK_IMAGE), fallback_img)
        timeline_valid = [{
            "timeframe": f"{sec_to_ts(0)} --> {sec_to_ts(audio_dur)}",
            "image_path": str(TMP_FALLBACK_IMAGE)
        }]

    # ================= PREFIX (MUTED) + VOICEOVER STARTS AT 0 =================
    if input_videos and input_videos_position == "prefix" and not effective_input_videos_audio:
        audio_dur = get_media_duration(audio_file)
        if audio_dur <= 0:
            audio_dur = None

        # Precompute valid input videos once
        video_items = []
        for item in input_videos:
            if isinstance(item, dict):
                video_path = item.get("path") or item.get("video_path")
            else:
                video_path = item

            if not video_path:
                continue

            video_path = Path(video_path)
            if not video_path.exists():
                continue

            duration = get_media_duration(video_path)
            if duration <= 0:
                continue
            video_items.append((video_path, duration))

        prefix_entries = []
        prefix_total = 0.0

        def add_prefix_entry(video_path, duration):
            nonlocal prefix_total
            start_ts = sec_to_ts(prefix_total)
            end_ts = sec_to_ts(prefix_total + duration)
            prefix_entries.append({
                "timeframe": f"{start_ts} --> {end_ts}",
                "video_path": str(video_path)
            })
            prefix_total += duration

        if audio_dur is not None and not timeline_valid:
            # Only video given and no photos: loop video(s) to match audio length
            if video_items:
                while prefix_total + 0.001 < audio_dur:
                    progressed = False
                    for video_path, duration in video_items:
                        remaining = audio_dur - prefix_total
                        if remaining <= 0:
                            break
                        use_dur = duration if duration <= remaining else remaining
                        if use_dur <= 0:
                            continue
                        add_prefix_entry(video_path, use_dur)
                        progressed = True
                        if prefix_total + 0.001 >= audio_dur:
                            break
                    if not progressed:
                        break
        else:
            for video_path, duration in video_items:
                if audio_dur is not None:
                    remaining = audio_dur - prefix_total
                    if remaining <= 0:
                        break
                    if duration > remaining:
                        duration = remaining
                add_prefix_entry(video_path, duration)

        combined = list(prefix_entries)
        if audio_dur is None:
            # Fallback to legacy behavior if audio duration is unavailable
            shifted_timeline = []
            for entry in timeline_valid:
                start, end = parse_timeframe(entry["timeframe"])
                start += prefix_total
                end += prefix_total
                new_entry = dict(entry)
                new_entry["timeframe"] = f"{sec_to_ts(start)} --> {sec_to_ts(end)}"
                shifted_timeline.append(new_entry)
            combined += shifted_timeline
        else:
            remaining = max(0.0, audio_dur - prefix_total)
            if remaining > 0:
                trimmed_timeline, _ = build_sequential_entries(
                    timeline_valid,
                    max_duration=remaining,
                    offset=prefix_total
                )
                combined += trimmed_timeline

        render_timeline_video(
            timeline=combined,
            words=words,
            output_path=TMP_MAIN_VIDEO,
            template=template,
            heading_text=heading_text,
            render_subtitles=subtitles_enabled
        )

        total_video_dur = get_media_duration(TMP_MAIN_VIDEO)
        target_dur = audio_dur if audio_dur is not None else total_video_dur
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(TMP_MAIN_VIDEO),
            "-i", str(audio_file),
            "-filter_complex", f"[1:a]atrim=0:{target_dur}[a]",
            "-map", "0:v:0",
            "-map", "[a]",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-t", str(target_dur),
            str(output_video)
        ])

        if not effective_input_videos_audio:
            apply_default_audio_after_voiceover(output_video, audio_file, default_audio)

        # Cleanup temp files
        for fpath in [TMP_MAIN_VIDEO, TMP_MAIN_WITH_AUDIO, TMP_VIDEO]:
            if os.path.exists(fpath):
                os.remove(fpath)
        print(f"DONE: {output_video}")
        return

    # ================= MAIN RENDER =================
    render_timeline_video(
        timeline=timeline_valid,
        words=words,
        output_path=TMP_MAIN_VIDEO,
        template=template,
        heading_text=heading_text,
        render_subtitles=subtitles_enabled
    )

    # Mux main audio (voiceover)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(TMP_MAIN_VIDEO),
        "-i", str(audio_file),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-c:a", "aac",
        str(TMP_MAIN_WITH_AUDIO)
    ])

    # ================= INPUT VIDEOS (PREFIX/SUFFIX) =================
    segment_files = []
    segment_durations = []
    if input_videos:
        for i, item in enumerate(input_videos):
            if isinstance(item, dict):
                video_path = item.get("path") or item.get("video_path")
                video_audio = item.get("video_audio", input_videos_audio)
            else:
                video_path = item
                video_audio = input_videos_audio

            if not video_path:
                continue

            video_path = Path(video_path)
            if not video_path.exists():
                continue

            duration = get_media_duration(video_path)
            if duration <= 0:
                continue

            seg_timeline = [{
                "timeframe": f"{sec_to_ts(0)} --> {sec_to_ts(duration)}",
                "video_path": str(video_path)
            }]

            seg_video = BASE_DIR / f"_tmp_input_{i}.mp4"
            render_timeline_video(
                timeline=seg_timeline,
                words=None,
                output_path=seg_video,
                template=template,
                heading_text=heading_text,
                render_subtitles=False
            )

            seg_with_audio = BASE_DIR / f"_tmp_input_{i}_with_audio.mp4"
            if video_audio:
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(seg_video),
                    "-i", str(video_path),
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    str(seg_with_audio)
                ])
            else:
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(seg_video),
                    "-f", "lavfi",
                    "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-shortest",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    str(seg_with_audio)
                ])

            segment_files.append(seg_with_audio)
            segment_durations.append(duration)

    # ================= FINAL OUTPUT =================
    if segment_files:
        if input_videos_position not in ("prefix", "suffix"):
            input_videos_position = "prefix"

        ordered = (
            segment_files + [TMP_MAIN_WITH_AUDIO]
            if input_videos_position == "prefix"
            else [TMP_MAIN_WITH_AUDIO] + segment_files
        )

        # If prefix videos are muted, voiceover must start at t=0.
        if input_videos_position == "prefix" and not effective_input_videos_audio:
            # concat video-only, then mux voiceover starting at 0
            cmd = ["ffmpeg", "-y"]
            for seg in ordered:
                cmd += ["-i", str(seg)]
            concat_inputs = "".join([f"[{i}:v]" for i in range(len(ordered))])
            cmd += [
                "-filter_complex", f"{concat_inputs}concat=n={len(ordered)}:v=1:a=0[v]",
                "-map", "[v]",
                "-c:v", "libx264",
                str(TMP_CONCAT_VIDEO)
            ]
            subprocess.run(cmd)

            total_video_dur = get_media_duration(TMP_CONCAT_VIDEO)
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(TMP_CONCAT_VIDEO),
                "-i", str(audio_file),
                "-filter_complex", f"[1:a]apad,atrim=0:{total_video_dur}[a]",
                "-map", "0:v:0",
                "-map", "[a]",
                "-c:v", "libx264",
                "-c:a", "aac",
                str(output_video)
            ])
        else:
            # concat with audio+video
            cmd = ["ffmpeg", "-y"]
            for seg in ordered:
                cmd += ["-i", str(seg)]

            concat_inputs = "".join([f"[{i}:v][{i}:a]" for i in range(len(ordered))])
            cmd += [
                "-filter_complex", f"{concat_inputs}concat=n={len(ordered)}:v=1:a=1[v][a]",
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-c:a", "aac",
                str(output_video)
            ]
            subprocess.run(cmd)

            if not effective_input_videos_audio:
                apply_default_audio_after_voiceover(output_video, audio_file, default_audio)
    else:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(TMP_MAIN_WITH_AUDIO),
            "-c:v", "libx264",
            "-c:a", "aac",
            str(output_video)
        ])
        apply_default_audio_after_voiceover(output_video, audio_file, default_audio)

    # ================= CLEANUP =================
    for fpath in [
        TMP_VIDEO,
        TMP_MAIN_VIDEO,
        TMP_MAIN_WITH_AUDIO,
        TMP_PREFIX_SUFFIX,
        TMP_CONCAT_VIDEO,
        TMP_SUFFIX_AUDIO,
        TMP_FULL_AUDIO,
        TMP_FALLBACK_IMAGE
    ]:
        if os.path.exists(fpath):
            os.remove(fpath)

    if input_videos:
        for i in range(len(input_videos)):
            for suffix in ["", "_with_audio"]:
                p = BASE_DIR / f"_tmp_input_{i}{suffix}.mp4"
                if os.path.exists(p):
                    os.remove(p)

    print(f"DONE: {output_video}")

if __name__ == "__main__":
    create_video(
        TIMELINE_JSON,
        WORD_TIMELINE_JSON,
        AUDIO_FILE,
        OUTPUT_VIDEO,
        TEMPLATE_JSON,
        "UGC नियमों पर सुप्रीम कोर्ट की रोक: क्या बदलेगा अब?",
        input_videos=[
        {"path": "1212121212.mp4", "video_audio": False}
        ],
        input_videos_position="prefix",
        input_videos_audio=False,
        default_audio="/Users/pragyan/Voicecline/app/BreakinNews-2.mp3"
    )
