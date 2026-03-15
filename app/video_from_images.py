import json
import cv2
import numpy as np
import subprocess
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# USER INPUTS (ONLY THESE NEED TO BE CHANGED)
# ============================================================
SRT_FILE = "1769260471812.srt"
TIMELINE_JSON = "final_1769260471812.json"
AUDIO_FILE = "1769260471812.wav"
OUTPUT_VIDEO = "final_1769260471812video.mp4"

# ============================================================
# INTERNAL CONFIG
# ============================================================
FRAME_WIDTH = 1440
FRAME_HEIGHT = 1080
FPS = 30

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
def _resolve_font_path():
    candidates = [
        PROJECT_ROOT / "Noto_Sans_Devanagari-2" / "static" / "NotoSansDevanagari_Condensed-Bold.ttf",
        PROJECT_ROOT / "card-new" / "Font" / "Noto_Sans_Devanagari" / "static" / "NotoSansDevanagari-Regular.ttf",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])

FONT_PATH = _resolve_font_path()
FONT_SIZE = 48
TEXT_BG_COLOR = (0, 0, 0, 180)
TEXT_COLOR = (255, 255, 255, 255)

TMP_SRT = "_fixed_tmp.srt"
TMP_VIDEO = "_tmp_video.mp4"

print("▶ Starting video build")

# ============================================================
# AUDIO DURATION
# ============================================================
def get_audio_duration(audio_path):
    r = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         audio_path],
        stdout=subprocess.PIPE,
        text=True
    )
    return float(r.stdout.strip())

AUDIO_DURATION = get_audio_duration(AUDIO_FILE)
print(f"🎧 Audio duration: {AUDIO_DURATION:.2f}s")

# ============================================================
# FIX SRT
# ============================================================
def fix_srt(src, dst, audio_duration):
    def fix_ts(ts):
        h, m, rest = ts.split(":")
        s, ms = rest.split(",")

        sec = int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
        if sec > audio_duration + 0.5:
            return f"00:{int(h):02d}:{int(m):02d},{int(ms):03d}"
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(ms):03d}"

    out = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            if "-->" in line:
                a, b = line.strip().split("-->")
                out.append(f"{fix_ts(a.strip())} --> {fix_ts(b.strip())}\n")
            else:
                out.append(line)

    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(out)

fix_srt(SRT_FILE, TMP_SRT, AUDIO_DURATION)

# ============================================================
# SRT PARSER
# ============================================================
def parse_srt(path):
    subs, block = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if len(block) >= 3:
                    subs.append({
                        "timeframe": block[1],
                        "text": " ".join(block[2:])
                    })
                block = []
            else:
                block.append(line)
    return subs

def ts_to_seconds(ts):
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def parse_timeframe(tf):
    s, e = tf.split("-->")
    return ts_to_seconds(s.strip()), ts_to_seconds(e.strip())

# ============================================================
# TEXT RENDER
# ============================================================
def render_text_image(text):
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE, layout_engine=ImageFont.Layout.RAQM)
    except OSError:
        font = ImageFont.load_default()
    max_width = FRAME_WIDTH - 100

    words, lines, cur = text.split(), [], ""
    dummy = Image.new("RGB", (FRAME_WIDTH, 100))
    d = ImageDraw.Draw(dummy)

    for w in words:
        t = cur + (" " if cur else "") + w
        if d.textlength(t, font=font) <= max_width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    lh = FONT_SIZE + 10
    h = lh * len(lines) + 40
    img = Image.new("RGBA", (FRAME_WIDTH, h), TEXT_BG_COLOR)
    d = ImageDraw.Draw(img)

    y = 20
    for line in lines:
        w = d.textlength(line, font=font)
        d.text(((FRAME_WIDTH - w)//2, y), line, font=font, fill=TEXT_COLOR)
        y += lh

    return np.array(img)

# ============================================================
# LOAD INPUT DATA
# ============================================================
with open(TIMELINE_JSON, "r", encoding="utf-8") as f:
    timeline = json.load(f)

subs = parse_srt(TMP_SRT)
count = min(len(subs), len(timeline))
print(f"▶ Rendering {count} segments")

# ============================================================
# VIDEO WRITER
# ============================================================
writer = cv2.VideoWriter(
    TMP_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (FRAME_WIDTH, FRAME_HEIGHT)
)

current_time = 0.0
last_frame = None

# ============================================================
# RENDER LOOP
# ============================================================
for i in range(count):
    start, end = parse_timeframe(subs[i]["timeframe"])
    duration = end - start
    if duration <= 0:
        continue

    entry = timeline[i]
    img_path = (
        entry.get("image_or_video_path")
        or entry.get("image_path")
    )

    if not img_path or not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    bg = cv2.GaussianBlur(
        cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT)),
        (51, 51), 0
    )

    h, w, _ = img.shape
    ar_i = w / h
    ar_f = FRAME_WIDTH / FRAME_HEIGHT
    nw, nh = (
        (FRAME_WIDTH, int(FRAME_WIDTH / ar_i))
        if ar_i > ar_f else
        (int(FRAME_HEIGHT * ar_i), FRAME_HEIGHT)
    )

    fg = cv2.resize(img, (nw, nh))
    x = (FRAME_WIDTH - nw)//2
    y = (FRAME_HEIGHT - nh)//2
    bg[y:y+nh, x:x+nw] = fg

    text_img = render_text_image(subs[i]["text"])
    th = text_img.shape[0]
    a = text_img[:, :, 3] / 255.0
    for c in range(3):
        bg[FRAME_HEIGHT-th:FRAME_HEIGHT, :, c] = (
            a * text_img[:, :, c] +
            (1 - a) * bg[FRAME_HEIGHT-th:FRAME_HEIGHT, :, c]
        )

    frames = int(round(duration * FPS))
    for _ in range(frames):
        writer.write(bg)

    last_frame = bg
    current_time = end

writer.release()

# ============================================================
# AUDIO MUX
# ============================================================
subprocess.run([
    "ffmpeg", "-y",
    "-i", TMP_VIDEO,
    "-i", AUDIO_FILE,
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-c:v", "libx264",
    "-c:a", "aac",
    OUTPUT_VIDEO
], check=True)

os.remove(TMP_VIDEO)
os.remove(TMP_SRT)

print("✔ DONE")
