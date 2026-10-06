import re
import pandas as pd
from pathlib import Path

# === CONFIG ===
INPUT_CSV  = r"C:\Users\AhmedYasser\Documents\Personal\Heriott_Watt_MSc\F21DL-Machine_Learning\Project\F21DL-Project\Dataset\Mohamed_Dataset\snapshots\youtube_dataset_snapshot_20251104_144630.csv"
OUTPUT_CSV = r"C:\Users\AhmedYasser\Documents\Personal\Heriott_Watt_MSc\F21DL-Machine_Learning\Project\F21DL-Project\Dataset\Mohamed_Dataset\snapshots\youtube_dataset_filtered.csv"

# --- Helpers ---

ISO8601_DUR_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$",
    re.IGNORECASE
)

def parse_iso8601_duration_to_seconds(s: str) -> float:
    if not isinstance(s, str):
        return float('nan')
    s = s.strip()
    m = ISO8601_DUR_RE.match(s)
    if not m:
        return float('nan')
    days = int(m.group('days') or 0)
    hours = int(m.group('hours') or 0)
    minutes = int(m.group('minutes') or 0)
    seconds = int(m.group('seconds') or 0)
    return days*86400 + hours*3600 + minutes*60 + seconds

def first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_duration_seconds(df):
    duration_cols = [
        "duration_seconds", "durationSecs", "lengthSeconds",
        "duration", "videoDuration", "contentDetails.duration"
    ]
    col = first_existing_column(df, duration_cols)
    if col is None:
        return pd.Series([float('nan')] * len(df), index=df.index)
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.5:
        return s_num.astype(float)
    return s.apply(parse_iso8601_duration_to_seconds)

def build_text_field(df):
    candidates = ["title", "videoTitle", "name",
                  "description", "videoDescription",
                  "tags", "videoTags"]
    parts = []
    for c in candidates:
        if c in df.columns:
            parts.append(df[c].fillna("").astype(str))
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    return pd.concat(parts, axis=1).agg(" ".join, axis=1)

def has_thumbnail_mask(df):
    thumb_cols = ["thumbnail_file", "thumbnail", "thumbnail_url", "thumbnailUrl", "thumbnailPath"]
    col = first_existing_column(df, thumb_cols)
    if col is None:
        return pd.Series([False] * len(df), index=df.index)
    s = df[col].astype(str).str.strip()
    bad = s.eq("") | s.str.lower().eq("nan") | s.eq("#NAME?")
    return ~bad

def coerce_views(df):
    view_cols = ["viewCount", "views", "statistics.viewCount", "ViewCount"]
    col = first_existing_column(df, view_cols)
    if col is None:
        return pd.Series([float('nan')]*len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")

def is_youtube_short_mask(text, duration_seconds):
    hashtag_re = re.compile(r"(#shorts\b|#fyp\b)", re.IGNORECASE)
    has_hashtag = text.str.contains(hashtag_re, na=False)
    too_short = duration_seconds < 30
    return has_hashtag | too_short

# === Load ===
df = pd.read_csv(INPUT_CSV)
orig_n = len(df)

# --- Build helper fields ---
duration_seconds = ensure_duration_seconds(df)
text_all = build_text_field(df)
views = coerce_views(df)

# --- Masks ---
mask_has_thumb = has_thumbnail_mask(df)
mask_not_short = ~is_youtube_short_mask(text_all, duration_seconds)
mask_views_ok = views >= 100

# --- Apply filters ---
filtered = df[mask_has_thumb & mask_not_short & mask_views_ok].copy()

# --- Reporting ---
print(f"Rows in:  {orig_n}")
print(f"  - Missing/invalid thumbnail removed: {(~mask_has_thumb).sum()}")
print(f"  - YouTube Shorts removed: {(~mask_not_short).sum()}")
print(f"  - <100 views removed: {(~mask_views_ok).sum()}")
print(f"Rows out: {len(filtered)}")

# --- Save ---
filtered.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
