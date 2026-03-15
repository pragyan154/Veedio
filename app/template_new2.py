import cv2
import json
import numpy as np
import subprocess
import os
from PIL import Image, ImageDraw, ImageFont
import textwrap
from pathlib import Path

# ================= FILES =================
TIMELINE_JSON = "/Users/pragyan/Voicecline/app/final_1769603184754.json"                 # image + timeframe mapping
WORD_TIMELINE_JSON = "/Users/pragyan/Voicecline/app/1769603184754_wordscaptions"   # word-level timestamps
TEMPLATE_JSON = "template.json"
AUDIO_FILE = "/Users/pragyan/Voicecline/app/1769603184754.wav"
OUTPUT_VIDEO = "finaaaassss_1769603184754.mp4"
TMP_VIDEO = "_tmp.mp4"

# ================= FONT =================
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
FONT_SIZE = 62

def _get_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()

SUB_BG = (0, 0, 0, 180)
SUB_TEXT = (200, 200, 200, 255)
SUB_ACTIVE = (255, 220, 80, 255)

SUB_PADDING = 30
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

def render_word_window(word_list, active_idx, max_width):
    font = _get_font(FONT_SIZE)

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
            d.text((x, y), w, font=font, fill=color)
            x += widths[wi] + space
            word_counter += 1

        y += line_heights[li] + line_gap

    return np.array(img)


def render_heading(text, max_width, heading_cfg):
    if not text:
        return None

    font_size = heading_cfg.get("font_size", 50)
    font = _get_font(font_size)
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

# ================= FUNCTIONAL ENTRYPOINT =================
def create_video(timeline_json, word_timeline_json, audio_file, output_video, template_json="template.json", heading_text=""):
    # ================= LOAD DATA =================
    with open(template_json, "r") as f:
        template = json.load(f)

    with open(timeline_json, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    with open(word_timeline_json, "r", encoding="utf-8") as f:
        words = json.load(f)

    if not words:
        raise ValueError("Word timeline empty")

    WIDTH = template["canvas"]["width"]
    HEIGHT = template["canvas"]["height"]
    FPS = template["canvas"]["fps"]

    BG_CFG = template.get("background", {})
    MEDIA_BOX = template["main_media"]
    SUB_CFG = template["subtitle"]
    HEADING_CFG = template.get("heading", {})

    # If heading_text is not provided, try to get it from template
    if not heading_text:
        heading_text = HEADING_CFG.get("text", "")

    # ================= VIDEO WRITER =================
    writer = cv2.VideoWriter(
        TMP_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT)
    )

    # ================= BACKGROUND INIT =================
    bg_img = None
    bg_cap = None
    bg_src = BG_CFG.get("source_from")

    if bg_src and os.path.exists(bg_src):
        if bg_src.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            bg_cap = cv2.VideoCapture(bg_src)
        else:
            bg_img = cv2.imread(bg_src)

    # ================= MEDIA BOX =================
    bw = int(WIDTH * MEDIA_BOX["w"])
    bh = int(HEIGHT * MEDIA_BOX["h"])
    bx = int(WIDTH * MEDIA_BOX["x"])
    by = int(HEIGHT * MEDIA_BOX["y"])

    # ================= RENDER =================
    word_idx = 0
    window_start_idx = 0
    total_words = len(words)

    # Pre-render heading
    heading_layer = render_heading(heading_text, WIDTH, HEADING_CFG)

    for entry in timeline:
        start, end = parse_timeframe(entry["timeframe"])
        start_f = int(start * FPS)
        end_f = int(end * FPS)

        main_img = cv2.imread(entry["image_path"])
        if main_img is None:
            continue

        for frame_idx in range(start_f, end_f):
            t = frame_idx / FPS

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
            frame_box = blur_fill_from_media(main_img, bw, bh)
            media = fit_contain(main_img, bw, bh)
            mx = (bw - media.shape[1]) // 2
            my = (bh - media.shape[0]) // 2
            frame_box[my:my+media.shape[0], mx:mx+media.shape[1]] = media
            bg[by:by+bh, bx:bx+bw] = frame_box

            # ---- HEADING ----
            if heading_layer is not None:
                hh = heading_layer.shape[0]
                hy = HEADING_CFG.get("y_margin_top", 50)

                # RGBA → BGR
                heading_rgb = heading_layer[:, :, :3][:, :, ::-1]
                h_alpha = heading_layer[:, :, 3] / 255.0

                for c in range(3):
                    bg[hy:hy+hh, :, c] = (
                        h_alpha * heading_rgb[:, :, c] +
                        (1 - h_alpha) * bg[hy:hy+hh, :, c]
                    )

            # ---- SUBTITLE ----
            sub_img = render_word_window(window_words, active_local_idx, WIDTH)
            sh = sub_img.shape[0]
            sy = HEIGHT - sh - SUB_CFG["y_margin_bottom"]
            alpha = sub_img[:, :, 3] / 255.0

            for c in range(3):
                bg[sy:sy+sh, :, c] = (
                    alpha * sub_img[:, :, c] +
                    (1 - alpha) * bg[sy:sy+sh, :, c]
                )

            writer.write(bg)

    # ================= CLEANUP =================
    if bg_cap:
        bg_cap.release()

    writer.release()

    subprocess.run([
        "ffmpeg", "-y",
        "-i", TMP_VIDEO,
        "-i", audio_file,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-c:a", "aac",
        output_video
    ])

    if os.path.exists(TMP_VIDEO):
        os.remove(TMP_VIDEO)
    print(f"DONE: {output_video}")

if __name__ == "__main__":
    create_video(TIMELINE_JSON, WORD_TIMELINE_JSON, AUDIO_FILE, OUTPUT_VIDEO, TEMPLATE_JSON , "बरेली: सिटी मजिस्ट्रेट का इस्तीफा, गणतंत्र दिवस पर प्रशासनिक हलचल")
