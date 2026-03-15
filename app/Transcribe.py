import os
from pathlib import Path
import io
import wave
from pydub import AudioSegment
from google.cloud import speech
import json
from text_utils import ensure_hinglish_roman

BASE_DIR = Path(__file__).resolve().parent
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(BASE_DIR / "scibugai-0c6edfa92c76.json")


def format_timestamp(seconds):
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    millis = int((seconds % 1) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

def get_wav_metadata(file_path):
    with wave.open(file_path, "rb") as wav_file:
        return wav_file.getframerate(), wav_file.getnchannels()

def transcribe_to_srt_limited(
    input_wav,
    output_srt,
    output_words,
    language="hindi",
):
    language_norm = (language or "hindi").strip().lower()
    sample_rate, channels = get_wav_metadata(input_wav)
    audio = AudioSegment.from_wav(input_wav)
    total_ms = len(audio)

    if language_norm == "hinglish":
        lang_code = "en-IN"
        alt_codes = []
    else:
        lang_code = "hi-IN"
        alt_codes = ["en-IN"]

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        audio_channel_count=channels,
        language_code=lang_code,
        alternative_language_codes=alt_codes,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    )

    # Use chunking but WITHOUT overlap to avoid duplicate words and sync issues
    # Long running recognize with inline audio has a 1-minute limit.
    chunk_length_ms = 59 * 1000  # Keep chunks just under 1 minute
    start_ms = 0
    word_timeline = []

    print(f"Starting chunked transcription (Total: {total_ms/1000:.1f}s)...")
    
    while start_ms < total_ms:
        end_ms = min(start_ms + chunk_length_ms, total_ms)
        chunk = audio[start_ms:end_ms]

        buf = io.BytesIO()
        chunk.export(buf, format="wav")
        content = buf.getvalue()

        audio_data = speech.RecognitionAudio(content=content)
        
        # Using recognize instead of long_running_recognize for chunks < 1 min
        response = client.recognize(config=config, audio=audio_data)

        for result in response.results:
            alternative = result.alternatives[0]
            for w in alternative.words:
                ws = w.start_time.total_seconds() + (start_ms / 1000.0)
                we = w.end_time.total_seconds() + (start_ms / 1000.0)
                word_text = w.word
                if language_norm == "hinglish":
                    word_text = ensure_hinglish_roman(word_text)
                word_obj = {
                    "word": word_text,
                    "start": round(ws, 3),
                    "end": round(we, 3)
                }
                word_timeline.append(word_obj)
        
        start_ms = end_ms

    print("Transcription complete.")

    final_srt_lines = []
    block_id = 1

    # Re-grouping words into SRT blocks from the master word_timeline
    current_group = []
    for word in word_timeline:
        current_group.append(word)
        duration = current_group[-1]["end"] - current_group[0]["start"]

        if len(current_group) >= 14 or duration >= 8.0:
            s_t = format_timestamp(current_group[0]["start"])
            e_t = format_timestamp(current_group[-1]["end"])
            txt = " ".join(w["word"] for w in current_group)
            if language_norm == "hinglish":
                txt = ensure_hinglish_roman(txt)
            final_srt_lines.append(f"{block_id}\n{s_t} --> {e_t}\n{txt}\n\n")
            block_id += 1
            current_group = []
    
    # Add remaining words
    if current_group:
        s_t = format_timestamp(current_group[0]["start"])
        e_t = format_timestamp(current_group[-1]["end"])
        txt = " ".join(w["word"] for w in current_group)
        if language_norm == "hinglish":
            txt = ensure_hinglish_roman(txt)
        final_srt_lines.append(f"{block_id}\n{s_t} --> {e_t}\n{txt}\n\n")

    with open(output_srt, "w", encoding="utf-8") as f:
        f.writelines(final_srt_lines)

    with open(output_words, "w", encoding="utf-8") as f:
        json.dump(word_timeline, f, ensure_ascii=False, indent=2)

    print(f"SRT: {output_srt}")
    print(f"Word timeline: {output_words}")
