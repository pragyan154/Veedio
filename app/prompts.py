# prompts.py
# ============================================================
# Centralized prompt templates for the entire pipeline.
# All AI-facing prompts live here for easy editing and review.
# ============================================================


# ---------- voicecreate.py — Script generation ----------
def get_script_generation_prompt(content: str, min_seconds: int, max_seconds: int, heading_count: int, language: str = "hindi") -> str:
    """
    Build the script + heading generation prompt.
    `language` can be "hindi" or "hinglish".
    """
    if language == "hinglish":
        language_rules = (
            "- Use Hinglish (Hindi in English/Roman script with natural English mix).\n"
            "- Write ALL transcript and headings in Roman script only (no Devanagari).\n"
            "- Keep the tone conversational and natural, as if a young news anchor is speaking.\n"
            "- Do NOT force-translate English terms that are commonly used in English (e.g., 'Supreme Court', 'Parliament', 'social media', 'government', 'report').\n"
        )
        language_label = "Hinglish"
    else:
        language_rules = (
            "- Neutral, precise Hindi. No filler, no repetition.\n"
        )
        language_label = "Hindi"

    return f"""
Generate a concise, precise factual {language_label} news audio-ready transcript and {heading_count} short separate headings based strictly on the content below.

Rules:
- Output TWO parts only: Headings and Transcript.
- Headings:
  - Provide EXACTLY {heading_count} headings.
  - Each heading is short, factual {language_label}, 6–10 words, states the core update only.
  - Keep headings in chronological/story order matching the transcript flow.
- Transcript:
  - Single continuous transcript only.
  - NO headings, labels, or sections inside the transcript.
  - First 1–2 sentences must be a strong spoken hook (natural, not hype). Start the transcript immediately with the lead fact; do not use greetings or sign-offs.
  - Immediately convey why this news matters now.
  {language_rules.strip()}
  - When the content allows, weave ONE subtle, generic news-style interrogative phrase into the natural flow of the transcript for engagement, without framing it as a standalone question or adding new facts, speculation, or opinion.
  - Target duration: STRICTLY between {min_seconds} and {max_seconds} seconds.

OUTPUT FORMAT (STRICT):
HEADINGS:
1. <{language_label} Heading>
2. <{language_label} Heading>
...
TRANSCRIPT: <{language_label} Transcript>

Content:
{content}
""".strip()


# ---------- DownloadImage.py — Search query generation ----------
def get_search_query_prompt(subtitle_blocks_text: str) -> str:
    return f"""
You are a professional news video editor mapping subtitles to visuals.

Each subtitle block represents ONE video frame.
Generate EXACTLY ONE minimal image search query per subtitle block.

⚠️ CRITICAL SAFETY & NEUTRALITY RULE (HIGHEST PRIORITY):
- You MUST NOT generate any image query that identifies or implies:
  - Any specific person (name, face, role, title)
  - Any political figure, party, government body, election, or ideology
- Replace all such references with neutral, non-person visuals
  (crowd, location, building, documents, vehicles, symbols).

🚫 META VISUAL BAN (STRICT):
- You MUST NOT use meta or studio visuals such as:
  - news headline
  - breaking news
  - news graphic
  - studio shot
  - anchor desk
- This is a field-report style story, not a studio segment.

DECISION PRIORITY (STRICT ORDER):
1. Safety & neutrality (cannot be overridden)
2. If the subtitle explicitly mentions a SPECIFIC place or non-person object,
   ALWAYS use that exact entity.
3. If the subtitle introduces NO new visual entity,
   reuse the most recent story-relevant visual.
4. Context is used ONLY to maintain story continuity,
   never to invent new visuals.

EDITORIAL RULES:
- Treat the full SRT as ONE continuous news story.
- Maintain visual continuity like a real broadcast.
- Do NOT generalize when specificity exists.
- Do NOT introduce abstract or symbolic visuals unless already established.

SEARCH QUERY RULES:
- 2–3 words preferred (max 4).
- Concrete and commonly searchable.
- Place / object / environment focused.
- Hindi + English understanding required.
- Do NOT invent visuals.

OUTPUT FORMAT (STRICT):
- One line per subtitle block
- Each line MUST start with index in square brackets
- Output ONLY the image search query
- No punctuation
- No explanation

Subtitles:
{subtitle_blocks_text}
"""


# ---------- DownloadImage.py — Visual description generation ----------
def get_visual_description_prompt(subtitle_blocks_text: str) -> str:
    return f"""
You are generating precise visual descriptions for a news video.
The entire SRT represents ONE continuous news story.
Each subtitle corresponds to EXACTLY ONE video frame.

GENERAL CONTEXT:
• The story is primarily set in INDIA unless the SRT clearly indicates another country.
• Visuals must reflect realistic Indian environments, infrastructure, architecture, lighting, clothing, vehicles, and surroundings.

FOR EACH SUBTITLE:
1. Use ONLY information present in the current subtitle and earlier subtitles.
2. Identify the SINGLE most important subject or action being discussed.
3. Generate ONE detailed, concrete visual description that clearly depicts that subject.
4. Use a REAL-WORLD LOCATION that is explicitly mentioned or strongly implied in the SRT.
5. Ensure strict continuity with previously described visuals (same place, time, mood, progression).

VISUAL RULES (STRICT):
• NO abstract, symbolic, conceptual, or metaphorical visuals.
• NO random crowds, parks, skylines, or filler imagery.
• NO unrelated stock visuals.
• If a law, institution, building, infrastructure, or event is mentioned, show its REAL physical setting.
• People should appear ONLY if the subtitle explicitly discusses people or human actions.
• When people appear, show them by role or activity — never by name.

POLITICAL & PUBLIC FIGURE RULES (VERY IMPORTANT):
• Do NOT show real photographs or realistic depictions of political leaders.
• If a political leader, public figure, or political party is mentioned:
  → They MUST be shown ONLY as an ANIMATED or ILLUSTRATED representation.
  → NEVER show multiple political leaders together.
  → NEVER show party symbols, flags, election logos, or rallies.
• If politics is mentioned but no individual is required visually, prefer showing locations, institutions, or environmental context instead.

TEXT IN VISUALS:
• Avoid readable text whenever possible.
• If text appears (signboards, documents, screens, UI):
  → Ensure correct spelling.
  → Otherwise blur, crop, or obscure the text.

IMAGE STYLE & CLARITY:
• Description must be visually unambiguous and image-generator friendly.
• Clearly mention:
  → Location (city/state/place)
  → Indoor or outdoor setting
  → Time of day (if implied)
  → Key objects, actions, and spatial layout
• Do NOT mention camera terms, lenses, or cinematography unless required.

OUTPUT FORMAT (STRICT):
• One output line per subtitle.
• Each line must start with the subtitle index in square brackets.

Subtitles:
{subtitle_blocks_text}
"""


# ---------- DownloadImage.py — Regenerate missing descriptions ----------
def get_regenerate_descriptions_prompt(subtitle_blocks_text: str) -> str:
    return f"""
You are selecting what should be visually shown for a news video.
The entire SRT is one continuous news story.
Each subtitle corresponds to one video frame.
For each subtitle:
Use ONLY the information present in the SRT (and earlier subtitles).
Identify the main subject being discussed.
Write a detailed, concrete visual description that clearly represents that subject.
Use a real-world location that is directly related to the story and already implied or stated in the SRT.
Stay fully consistent with previously established context from the SRT.
Rules:
Do NOT use generic, symbolic, or abstract visuals.
Do NOT introduce random people, crowds, parks, or unrelated places.
If an institution, place, law, or event is mentioned, show its actual real-world setting.
People should appear ONLY if the subtitle is explicitly about people.
Avoid showing readable text and real person images whenever possible; instead, convey the information through environment, actions, objects, and storyline-driven visuals that reflect the narrative context.
No political parties, political symbols, or real political leaders should be shown visually.
Do NOT explicitly mention names of individuals in visual descriptions; represent people only by their roles or through storyline-based context, especially when using animated or illustrated visuals.
If any visual contains text (signboards, screens, documents, headlines, UI, etc.), ensure the text is correctly spelled; otherwise, the text must be blurred, obscured, or cropped out.
Every visual description must explicitly mention the real-world location and surrounding context so the environment clearly indicates where the image is from.
STRICT EXCEPTION:
If a specific political leader, public figure, or political party is named,
they must be shown ONLY as an ANIMATED or ILLUSTRATED representation.
No real photographs, realistic depictions, or multiple political figures together are allowed.
OUTPUT FORMAT (STRICT):
One output for EACH subtitle in the SRT.
Each line starts with the subtitle index in [ ].

Subtitles:
{subtitle_blocks_text}
""".strip()


# ---------- Imagen2.py — Image evaluation ----------
STRICT_MATCH_PROMPT = """
Subtitle text:
{sub_text}

Expected visual description:
{description}

Output STRICT JSON ONLY:
{{"usable": true|false, "score": number}}
"""


# ---------- ImageAnal.py — Input image analysis ----------
DEFAULT_IMAGE_ANALYSIS_PROMPT = "Provide a single-line, highly detailed factual summary of this image,including the names of any famous people or landmarks present."


# ---------- input_image_json_map.py — Image-to-subtitle mapping ----------
def get_image_mapping_prompt(image_block: str, subtitle_block: str, only_input_image: bool = False) -> str:
    if only_input_image:
        mapping_mode_rules = """- EVERY SINGLE SUBTITLE index MUST be assigned EXACTLY ONE image from the [INPUT — IMAGES] list.
- EVERY IMAGE from the [INPUT — IMAGES] list MUST be used at least once.
- You MUST repeat images from the [INPUT — IMAGES] list as many times as necessary to cover all subtitles.
- No subtitle index can be left without an image.
- Mapping Logic: For each subtitle, select the most relevant image from the provided list. If multiple images are relevant, choose the one that maintains the best visual narrative flow."""
    else:
        mapping_mode_rules = """- You MUST return a mapping for EVERY image provided in the [INPUT — IMAGES] list.
- For each image, select EXACTLY ONE subtitle index.
- Mapping Logic:
    1. Look for an explicit textual mention of the image's primary entity in the subtitles.
    2. If no explicit match exists, map it to the subtitle that is the closest semantically or provides the best context for that image.
    3. Ensure images are mapped in chronological order relative to the subtitles."""

    return f"""
[SYSTEM ROLE]
You are a deterministic data-mapping engine. 
No creativity. No inference beyond text.

[TASK]
You are given:
1) A list of IMAGES (each with a filename and visual description).
2) A list of SUBTITLES (each with an index and text).

Your goal is to map images to relevant subtitle indices.

[CRITICAL RULES — DO NOT VIOLATE]
{mapping_mode_rules}
- Use ALL subtitles as your search space.
- The image filename MUST be copied EXACTLY as provided.

[INPUT — IMAGES]
{image_block}

[INPUT — SUBTITLES]
{subtitle_block}

[OUTPUT FORMAT — STRICT JSON ONLY]
[
  {{
    "subtitle_index": <integer>,
    "image": "<exact filename>"
  }}
]
"""
