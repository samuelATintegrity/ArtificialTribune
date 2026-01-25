import json
import os
import re
import time
import hashlib
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import feedparser
import requests
from dotenv import load_dotenv

OUTPUT_DIR = "output"
LOOKBACK_HOURS = 36

TRACKING_PARAMS = {"fbclid", "gclid", "ref"}
TRACKING_PREFIXES = ("utm_",)

LIVE_UPDATE_PATTERNS = [
    r"\blive updates\b",
    r"\blive blog\b",
    r"\blive\b",
]

PLAN = {
    "TOP": {"min": 5, "max": 5},
    "US_POLITICS": {"min": 1, "max": 2},
    "FOREIGN_POLICY": {"min": 1, "max": 2},
    "WORLD": {"min": 1, "max": 2},
    "BUSINESS": {"min": 1, "max": 2},
    "TECH": {"min": 1, "max": 2},
    "SPORTS": {"min": 1, "max": 1},
}

def canonicalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        q = []
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            kl = k.lower()
            if kl in TRACKING_PARAMS:
                continue
            if any(kl.startswith(prefix) for prefix in TRACKING_PREFIXES):
                continue
            q.append((k, v))
        new_query = urlencode(q, doseq=True)
        return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, ""))  # drop fragment
    except Exception:
        return url

def normalize_title(title: str) -> str:
    t = title.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t

def stable_id(title: str, url: str) -> str:
    base = normalize_title(title) + "|" + canonicalize_url(url)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]

def looks_like_live_update(title: str) -> bool:
    t = title.lower()
    return any(re.search(pat, t) for pat in LIVE_UPDATE_PATTERNS)

def parse_entry_datetime(entry):
    tt = entry.get("published_parsed") or entry.get("updated_parsed")
    if not tt:
        return None
    return datetime.fromtimestamp(time.mktime(tt), tz=timezone.utc)

def load_categorized_feeds():
    feeds = []
    with open("feeds.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                raise ValueError(f"Bad line in feeds.txt (missing '|'): {line}")
            category, url = line.split("|", 1)
            feeds.append((category.strip(), url.strip()))
    return feeds

def fetch_items():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    feeds = load_categorized_feeds()

    raw = []
    for category, url in feeds:
        feed = feedparser.parse(url)
        for e in feed.entries:
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            published_dt = parse_entry_datetime(e)

            if not title or not link or not published_dt:
                continue
            if published_dt < cutoff:
                continue

            c_url = canonicalize_url(link)
            raw.append({
                "id": stable_id(title, c_url),
                "title": title,
                "url": c_url,
                "published_utc": published_dt.isoformat(),
                "category": category,
                "feed": url,
                "is_live_update": looks_like_live_update(title),
            })

    dedup = {}
    for it in raw:
        existing = dedup.get(it["id"])
        if not existing or it["published_utc"] > existing["published_utc"]:
            dedup[it["id"]] = it

    items = list(dedup.values())
    items.sort(key=lambda x: x["published_utc"], reverse=True)
    return raw, items

def pick_items(items, category, used_ids, count, avoid_live_updates=True):
    picks = []
    for it in items:
        if it["id"] in used_ids:
            continue
        if it["category"] != category:
            continue
        if avoid_live_updates and it.get("is_live_update"):
            continue
        picks.append(it)
        used_ids.add(it["id"])
        if len(picks) >= count:
            break
    return picks

def build_lineup(items):
    used_ids = set()
    lineup = {}

    lineup["TOP"] = pick_items(items, "TOP", used_ids, PLAN["TOP"]["max"], avoid_live_updates=True)

    for cat in ["US_POLITICS", "FOREIGN_POLICY", "WORLD", "BUSINESS", "TECH", "SPORTS"]:
        lineup[cat] = pick_items(items, cat, used_ids, PLAN[cat]["max"], avoid_live_updates=True)

        if len(lineup[cat]) < PLAN[cat]["min"]:
            more = pick_items(items, cat, used_ids, PLAN[cat]["min"] - len(lineup[cat]), avoid_live_updates=False)
            lineup[cat].extend(more)

    return lineup

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def openai_generate_script(lineup, model: str, api_key: str) -> str:
    """
    Uses OpenAI Responses API: POST https://api.openai.com/v1/responses
    Auth: Authorization: Bearer <key>
    """
    # Keep the model honest: do not invent facts beyond headlines.
    instructions = (
        "Write a daily news audio narration script.\n"
        "Constraints:\n"
        "- Use ONLY the provided headlines + URLs. Do NOT invent facts.\n"
        "- No duplicate stories across sections.\n"
        "- Output must be plain text suitable for ElevenLabs narration.\n"
        "- Length target: ~4 to 7 minutes.\n"
        "- Structure:\n"
        "  1) Cold open (1-2 sentences)\n"
        "  2) Top 5 stories (numbered)\n"
        "  3) US Politics (1-2)\n"
        "  4) Foreign Policy (1-2)\n"
        "  5) World (1-2)\n"
        "  6) Business/Economics (1-2)\n"
        "  7) Tech (1-2)\n"
        "  8) Sports (1)\n"
        "  9) Close (1 sentence)\n"
        "- At the very end, include a SOURCES section containing URLs only, one per line.\n"
        "- Avoid saying 'according to' repeatedly; vary phrasing.\n"
    )

    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": "Here is today's selected lineup as JSON:\n" + json.dumps(lineup, ensure_ascii=False)
            }
        ],
        "temperature": 0.4,
        # If you want to avoid server-side storage, uncomment the next line:
        # "store": False,
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()

    # Extract output text
    text_parts = []
    for out in data.get("output", []):
        for c in out.get("content", []):
            if c.get("type") == "output_text":
                text_parts.append(c.get("text", ""))

    script = "\n".join(text_parts).strip()
    if not script:
        raise RuntimeError("OpenAI returned no output_text.")
    return script

def main():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

    if not openai_key:
        raise SystemExit("Missing OPENAI_API_KEY in .env")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now().astimezone().strftime("%Y-%m-%d")

    raw, items = fetch_items()

    items_path = os.path.join(OUTPUT_DIR, f"{today}_items.json")
    write_json(items_path, items)

    lineup = build_lineup(items)
    lineup_path = os.path.join(OUTPUT_DIR, f"{today}_lineup.json")
    write_json(lineup_path, lineup)

    print(f"Wrote items:  {items_path}")
    print(f"Wrote lineup: {lineup_path}")

    # Generate the narration script (text only)
    script = openai_generate_script(lineup, model=model, api_key=openai_key)

    script_path = os.path.join(OUTPUT_DIR, f"{today}_script.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    print(f"Wrote script: {script_path}")
        # --- ElevenLabs: generate MP3 from the script ---
    eleven_key = os.getenv("ELEVEN_API_KEY", "").strip()
    voice_id = os.getenv("ELEVEN_VOICE_ID", "").strip()
    eleven_model = os.getenv("ELEVEN_MODEL_ID", "eleven_multilingual_v2").strip()

    if not eleven_key:
        raise SystemExit("Missing ELEVEN_API_KEY in .env")
    if not voice_id:
        raise SystemExit("Missing ELEVEN_VOICE_ID in .env")

    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": eleven_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": script,
        "model_id": eleven_model,
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.8
        }
    }

    r = requests.post(tts_url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()

    mp3_path = os.path.join(OUTPUT_DIR, f"{today}.mp3")
    with open(mp3_path, "wb") as f:
        f.write(r.content)

    print(f"Wrote MP3:   {mp3_path}")
        # --- Publish-ready copies (stable + dated) ---
    publish_dir = os.path.join(OUTPUT_DIR, "publish")
    os.makedirs(publish_dir, exist_ok=True)

    latest_path = os.path.join(publish_dir, "latest.mp3")
    dated_path = os.path.join(publish_dir, f"{today}.mp3")

    # Copy bytes (simple and reliable on Windows)
    with open(mp3_path, "rb") as src:
        audio_bytes = src.read()

    with open(latest_path, "wb") as dst:
        dst.write(audio_bytes)

    with open(dated_path, "wb") as dst:
        dst.write(audio_bytes)

    print(f"Publish latest: {latest_path}")
    print(f"Publish dated:  {dated_path}")



if __name__ == "__main__":
    main()
