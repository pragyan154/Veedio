# NewPick — AI Video Generation Pipeline

Automated Hindi/Hinglish news video generator. Takes text content, generates a scripted voiceover, transcribes to subtitles, finds/generates matching images, and renders a complete video with headings, logo, and background.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.11+ | Runtime |
| **uv** | latest | Fast package manager |
| **ffmpeg** | 6.0+ | Audio/video processing |
| **ffprobe** | (bundled with ffmpeg) | Media duration detection |

### Install prerequisites (macOS)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ffmpeg
brew install ffmpeg
```

---

## Quick Start

### 1. Clone & enter directory

```bash
cd /path/to/newpick
```

### 2. Create virtual environment with UV

```bash
uv venv .venv --python 3.11
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your API keys:

```env
# REQUIRED
GEMINI_API_KEY=your_gemini_api_key_here

# OPTIONAL (only if using OpenRouter)
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 6. Google Cloud Speech credentials

Place your service account JSON at:
```
app/scibugai-0c6edfa92c76.json
```
This is used by `Transcribe.py` for speech-to-text.

### 7. Run the pipeline

```bash
cd app
python main.py
```

---

## Docker UI (Minimal)

Build image:

```bash
docker build -t newspick-ui .
```

Run container:

```bash
docker run --rm -p 7860:7860 \
  -v "$(pwd)/app/InputImage:/app/app/InputImage" \
  -v "$(pwd)/app/final_videos:/app/app/final_videos" \
  newspick-ui
```

Open UI:

```
http://localhost:7860
```

Use the UI to set all runtime config and API keys, run pipeline, and watch progress logs.

---

## Configuration

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `GEMINI_API_KEY` | API key | *(required)* | Google Gemini API key |
| `OPENROUTER_API_KEY` | API key | *(empty)* | OpenRouter API key (optional) |
| `VIDEO_ORIENTATION` | `vertical`, `horizontal` | `vertical` | Video aspect ratio |
| `VIDEO_TEMPLATE_THEME` | `pg`, `failaan`, `newspick` (`np`) | `pg` | Visual theme |
| `LANGUAGE` | `hindi`, `hinglish` | `hindi` | Script generation language |

### config.json

Located at `app/config.json`. Key settings:

```json
{
    "ai_provider": "gemini",        // or "openrouter"
    "language": "hindi",            // or "hinglish"
    "text_models": ["gemini-2.5-flash", ...],
    "openrouter_text_models": ["google/gemini-2.5-flash-preview", ...],
    "max_retries": 5,
    "base_delay": 1.5
}
```

### Template Selection

Templates are auto-selected based on `VIDEO_ORIENTATION` × `VIDEO_TEMPLATE_THEME`:

| Orientation | Theme | Template File |
|-------------|-------|---------------|
| vertical | pg | `template.json` (1080×1350) |
| vertical | failaan | `template_failaan.json` (1080×1350) |
| vertical | newspick (`np`) | `template_newspick.json` (1080×1350) |
| horizontal | pg | `template_horizontal.json` (1920×1080) |
| horizontal | failaan | `template_failaan_horizontal.json` (1920×1080) |
| horizontal | newspick (`np`) | `template_newspick_horizontal.json` (1920×1080) |

---

## Usage Examples

### Generate a horizontal Hinglish video

```bash
VIDEO_ORIENTATION=horizontal LANGUAGE=hinglish python app/main.py
```

### Full sample: where to give all inputs

No separate script file is required. You provide raw news content, and the pipeline auto-generates the voice script.

1. Put your main news content in `app/main.py` inside `CONTENT = """ ... """`
2. Put optional input images in `app/InputImage/`
3. (Optional) Add intro/prefix videos in `app/main.py` under `input_videos = [...]`
4. Set runtime options using env vars

```bash
# from project root
VIDEO_ORIENTATION=vertical \
VIDEO_TEMPLATE_THEME=np \
LANGUAGE=hindi \
python app/main.py
```

Example content input location in `app/main.py`:

```python
CONTENT = """
यह आपका न्यूज़ कंटेंट है।
इसी से हेडिंग और वॉइसओवर स्क्रिप्ट ऑटो-जनरेट होगी।
"""
```

Optional prefix video sample in `app/main.py`:

```python
input_videos = [
    {"path": "intro.mp4", "video_audio": False}
]
input_videos_position = "prefix"
input_videos_audio = False
```

### Use OpenRouter instead of Gemini

1. Set `"ai_provider": "openrouter"` in `app/config.json`
2. Set `OPENROUTER_API_KEY` in `.env`
3. Run normally:
```bash
python app/main.py
```

### Logo Overlay

Place `logo.png` in the project root or `app/` directory. The logo will automatically appear at the position configured in the template JSON (default: `top-right`).

Supported positions: `top-left`, `top-right`, `bottom-left`, `bottom-right`

To change position, edit the `"logo"` section in the template JSON:
```json
"logo": {
    "enabled": true,
    "position": "top-right",
    "x_margin": 20,
    "y_margin": 20,
    "max_width": 120,
    "max_height": 60
}
```

---

## Pipeline Overview

```
Content Text
    │
    ├─ 1. Script Generation (voicecreate.py)
    │      Generates Hindi/Hinglish transcript + headings
    │
    ├─ 2. Audio Generation (voicecreate.py)
    │      Text-to-speech via Gemini TTS
    │
    ├─ 3. Transcription (Transcribe.py)
    │      Audio → SRT subtitles + word timeline
    │
    ├─ 4. Image Analysis (ImageAnal.py)
    │      Analyze input images (if provided)
    │
    ├─ 5. Image Mapping (input_image_json_map.py)
    │      Map input images to subtitle segments
    │
    ├─ 6. Search & Description (DownloadImage.py)
    │      Generate search queries + visual descriptions
    │
    ├─ 7. Image Resolution (Imagen2.py)
    │      Evaluate & generate images per subtitle
    │
    └─ 8. Video Rendering (template.py)
           Composite: bg + media + headings + subtitles + logo
           → final_videos/final_video_<run_id>.mp4
```

---

## Project Structure

```
newpick/
├── .env.example              # Environment variable template
├── .gitignore
├── requirements.txt          # Python dependencies (use with uv)
├── README.md
│
└── app/
    ├── main.py               # Pipeline orchestrator
    ├── prompts.py             # Centralized AI prompts
    ├── ai_utils.py            # Provider-aware AI helpers
    ├── openrouter_utils.py    # OpenRouter API integration
    ├── config.json            # Model & pipeline settings
    │
    ├── voicecreate.py         # Script + audio generation
    ├── Transcribe.py          # Speech-to-text (SRT)
    ├── ImageAnal.py           # Input image analysis
    ├── input_image_json_map.py # Image-subtitle mapping
    ├── DownloadImage.py       # Image search + download
    ├── Imagen2.py             # Image evaluation + generation
    │
    ├── template.py            # Video rendering engine
    ├── template.json          # Vertical PG template
    ├── template_failaan.json  # Vertical Failaan template
    ├── template_newspick.json # Vertical NewsPick template
    ├── template_horizontal.json        # Horizontal PG template
    ├── template_failaan_horizontal.json # Horizontal Failaan template
    ├── template_newspick_horizontal.json # Horizontal NewsPick template
    │
    ├── video_from_images.py   # Video-from-images helper
    ├── template_new2.py       # Alternative template renderer
    │
    ├── bg/                    # Background video assets
    ├── InputImage/            # User-provided input images
    ├── runs/                  # Per-run working directories
    └── final_videos/          # Output videos
```

---

## Troubleshooting

### "Missing GEMINI_API_KEY in environment"
→ Make sure you created `.env` from `.env.example` and filled in the key.

### ffmpeg not found
→ Install via `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux).

### Import errors when running
→ Make sure you activated the venv: `source .venv/bin/activate`

### Google Cloud Speech errors
→ Ensure `app/scibugai-0c6edfa92c76.json` exists with valid credentials.
