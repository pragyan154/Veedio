"""
User-editable pipeline inputs.
Update only this file, then run: python app/main.py
"""

# Main content input (paste your full story here)
CONTENT = """
Bhai, Splitsvilla X6 mein abhi tak ka sabse bada bawaal ho gaya hai – aur is baar drama sirf contestants ke beech nahi, balki khud host Sunny Leone aur contestant Yogesh Rawat ke beech chhid gaya hai!*
"""
# Script/audio behavior
AI_CONTENT = False  # False => use CONTENT exactly as-is for audio generation
VOICEOVER_SIZE = "large"  # small, medium, large, huge
VOICE_NAMES = [
"Zephyr",
"Puck",
"Charon",
"Kore",
"Fenrir",
"Leda",
"Orus",
"Aoede",
"Callirrhoe",
"Autonoe",
"Enceladus",
"Iapetus",
"Umbriel",
"Algieba",
"Despina",
"Erinome",
"Algenib",
"Rasalgethi",
"Laomedeia",
"Achernar",
"Alnilam",
"Schedar",
"Gacrux",
"Pulcherrima",
"Achird",
"Zubenelgenubi",
"Vindemiatrix",
"Sadachbia",
"Sadaltager",
"Sulafat"
]

# Language + visual mode
LANGUAGE = "hinglish"  # hindi, hinglish
VIDEO_ORIENTATION = "vertical"  # vertical, horizontal
INPUT_THEME = "newspick"  # pg, failaan, newspick, np

# Heading controls
ENABLE_HEADING = False
HEADING_INPUT = ""  # optional manual heading text/list; keep "" to auto-generate

# Subtitle controls
SUBTITLES_ENABLED = True

# Image pipeline controls
IMAGE_REVIEW_AI = False  # False => skip AI image analysis/review and use looping image timeline
IMAGE_LOOP_SECONDS = 5  # used when IMAGE_REVIEW_AI=False

# Optional input video controls (passed to create_video)
INPUT_VIDEOS = [
    # {"path": "1212121212.mp4", "video_audio": False}
]
INPUT_VIDEOS_POSITION = "prefix"  # prefix, suffix
INPUT_VIDEOS_AUDIO = False

# Optional background/default audio after voiceover
DEFAULT_AUDIO = "BreakinNews-2.mp3"  # relative to app/ or absolute path
