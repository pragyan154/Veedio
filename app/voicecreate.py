import os
import json
import time
import wave
import random
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from prompts import get_script_generation_prompt

# ---------- Config ----------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in environment.")

client = genai.Client(api_key=API_KEY)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
MINUTES = 2  # keep between 1–2
OUTPUT_WAV = "output.wav"


def load_config(path) -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        # First try relative to this file's directory
        candidate = BASE_DIR / cfg_path
        if candidate.exists():
            cfg_path = candidate
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config(CONFIG_PATH)

TEXT_MODELS = CONFIG.get("text_models", [])
VOICE_MODELS = CONFIG.get("voice_models", [])
VOICE_NAMES = CONFIG.get("voice_names", ["Schedar"])
MAX_RETRIES_PER_MODEL = int(CONFIG.get("max_retries", 5))
BASE_DELAY = float(CONFIG.get("base_delay", 1.5))

if not TEXT_MODELS:
    raise ValueError("config.json must include non-empty 'text_models'.")
if not VOICE_MODELS:
    raise ValueError("config.json must include non-empty 'voice_models'.")
if isinstance(VOICE_NAMES, str):
    VOICE_NAMES = [VOICE_NAMES]
if not VOICE_NAMES:
    VOICE_NAMES = ["Schedar"]


# ---------- Utils ----------
def save_wave_file(filename: str, pcm_data: bytes):
    with open(filename, "wb") as f:
        with wave.open(f, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm_data)


def parse_heading_and_transcript(text: str):
    """
    Accepts strict format:
      HEADINGS: ...
      TRANSCRIPT: ...
    But also tolerates minor drift / legacy single heading.
    """
    t = (text or "").strip()
    headings = []
    transcript = t

    def _parse_headings_block(block: str):
        items = []
        for raw in (block or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            # strip common list prefixes: "1.", "1)", "-", "*"
            if line[0].isdigit():
                line = line.lstrip("0123456789").lstrip(".)").strip()
            if line.startswith(("-", "*")):
                line = line[1:].strip()
            if line:
                items.append(line)
        return items

    if "HEADINGS:" in t:
        after_headings = t.split("HEADINGS:", 1)[1].strip()
        if "TRANSCRIPT:" in after_headings:
            h_block, tr = after_headings.split("TRANSCRIPT:", 1)
            headings = _parse_headings_block(h_block)
            transcript = tr.strip()
        elif "SCRIPT:" in after_headings:
            h_block, tr = after_headings.split("SCRIPT:", 1)
            headings = _parse_headings_block(h_block)
            transcript = tr.strip()
        else:
            # If no transcript delimiter, treat all lines as headings
            headings = _parse_headings_block(after_headings)
            transcript = ""
    elif "HEADING:" in t:
        after_heading = t.split("HEADING:", 1)[1].strip()
        # Prefer TRANSCRIPT: split; fall back to SCRIPT: if model used it
        if "TRANSCRIPT:" in after_heading:
            h, tr = after_heading.split("TRANSCRIPT:", 1)
            headings = [h.strip()] if h.strip() else []
            transcript = tr.strip()
        elif "SCRIPT:" in after_heading:
            h, tr = after_heading.split("SCRIPT:", 1)
            headings = [h.strip()] if h.strip() else []
            transcript = tr.strip()
        else:
            # If no transcript delimiter, treat first line as heading
            lines = after_heading.splitlines()
            heading = (lines[0] or "").strip()
            headings = [heading] if heading else []
            transcript = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    # Safety: if transcript still contains a label line, strip it
    if transcript.startswith("TRANSCRIPT:"):
        transcript = transcript.replace("TRANSCRIPT:", "", 1).strip()
    if transcript.startswith("SCRIPT:"):
        transcript = transcript.replace("SCRIPT:", "", 1).strip()

    return headings, transcript


def run_with_model_fallback(fn, models, *, retries_per_model: int, base_delay: float, label: str):
    """
    Priority logic requested:
      model[0] gets `retries_per_model` attempts,
      then model[1] gets `retries_per_model`, etc.
    """
    last_exc = None

    for model_index, model_name in enumerate(models):
        for attempt in range(retries_per_model):
            try:
                return fn(model_name)
            except Exception as e:
                last_exc = e
                wait = base_delay * (2 ** attempt)
                print(
                    f"[{label}] model[{model_index}]={model_name} "
                    f"attempt {attempt + 1}/{retries_per_model} failed: {e}"
                )
                # don't sleep after final attempt of final model
                if not (model_index == len(models) - 1 and attempt == retries_per_model - 1):
                    time.sleep(wait)

    raise last_exc


# ---------- Step 1: Generate structured script ----------
def generate_script(content: str, min_seconds: int, max_seconds: int, language: str = "hindi"):
    def _heading_count(min_s: int, max_s: int) -> int:
        if max_s <= 30:
            return 2
        if max_s <= 60:
            return 3
        return 4

    heading_count = _heading_count(min_seconds, max_seconds)
    prompt = get_script_generation_prompt(content, min_seconds, max_seconds, heading_count, language)

    # Check if we should use OpenRouter
    from ai_utils import get_ai_provider
    provider = get_ai_provider("script_generation")

    if provider == "openrouter":
        from ai_utils import smart_text_call
        raw_text = smart_text_call(prompt, task="script_generation")
        headings, transcript = parse_heading_and_transcript(raw_text)
        if not transcript:
            raise ValueError("Empty transcript produced.")
        return headings, transcript

    def _call(model_name: str):
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        headings, transcript = parse_heading_and_transcript(resp.text)
        if not transcript:
            raise ValueError("Empty transcript produced.")
        return headings, transcript

    return run_with_model_fallback(
        _call,
        TEXT_MODELS,
        retries_per_model=MAX_RETRIES_PER_MODEL,
        base_delay=BASE_DELAY,
        label="SCRIPT"
    )


# ---------- Step 2: Generate audio from script ----------
def generate_audio(script_text: str, output_wav_file: str, voice_names=None):
    candidate_voice_names = voice_names if voice_names is not None else VOICE_NAMES
    if isinstance(candidate_voice_names, str):
        candidate_voice_names = [candidate_voice_names]
    if not candidate_voice_names:
        candidate_voice_names = ["Schedar"]
    selected_voice = random.choice(candidate_voice_names)
    print(f"[TTS] Selected voice: {selected_voice}")

    def _call(model_name: str):
        resp = client.models.generate_content(
            model=model_name,
            contents=script_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=selected_voice
                        )
                    )
                )
            )
        )
        audio_data = resp.candidates[0].content.parts[0].inline_data.data
        if not audio_data:
            raise ValueError("No audio data returned.")
        save_wave_file(output_wav_file, audio_data)
        return output_wav_file

    return run_with_model_fallback(
        _call,
        VOICE_MODELS,
        retries_per_model=MAX_RETRIES_PER_MODEL,
        base_delay=BASE_DELAY,
        label="TTS"
    )


# ---------- Input ----------
CONTENT = """
    
""".strip()


# ---------- Run ----------
if __name__ == "__main__":
    min_seconds = max(10, int((MINUTES - 1) * 60))
    max_seconds = int(MINUTES * 60)
    headings, transcript = generate_script(CONTENT, min_seconds, max_seconds)
    print("\nHEADINGS:\n", headings)
    print("\nTRANSCRIPT (first 300 chars):\n", transcript[:300], "...\n")
    generate_audio(transcript, OUTPUT_WAV)
    print(f"Saved audio to: {OUTPUT_WAV}")
