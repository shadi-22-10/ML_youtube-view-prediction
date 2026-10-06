# AY_Pull_Channel_List.py â€” region-rotating, multi-seed, fixed scoring (playlist-free)
import os, csv, re, time, random, sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from itertools import cycle, islice
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json

API_KEY = os.getenv("YOUTUBE_API_KEY", "YOUR_API_KEY")

TARGETS = ["Tech","Gaming","Education","Music","Sports","Pets and Animals","Food and Cooking","Science"]

CATEGORY_SEEDS = {
    "Tech": ["tech","programming","robotics","electronics","AI","gadget review","pc build","coding tutorial"],
    "Gaming": ["gaming","gameplay","walkthrough","esports","speedrun","let's play"],
    "Education": ["lecture","tutorial","exam prep","math lesson","physics course","how to"],
    "Music": ["official video","cover song","remix","band live","dj set","guitar lesson"],
    "Sports": ["football highlights","basketball training","cricket match","tennis drills","ufc"],
    "Pets and Animals": ["dog training","cat care","wildlife","zoo vlog","veterinary"],
    "Food and Cooking": ["recipe","cooking","baking","chef tips","meal prep","kitchen hacks"],
    "Science": ["science experiment","biology lab","astronomy","chemistry","research talk"],
}

VALIDATION_KEYWORDS = {
    "Tech": {"tech","technology","gadget","electronics","programming","coding","developer","robot","robotics","ai","ml","review","pc"},
    "Gaming": {"game","gaming","gameplay","walkthrough","speedrun","esports","lets","let's","stream","twitch"},
    "Education": {"education","tutorial","lecture","course","lesson","how","class","exam"},
    "Music": {"music","official","lyrics","cover","remix","band","dj","orchestra","piano","guitar","track","album"},
    "Sports": {"sport","football","soccer","basketball","cricket","highlights","tennis","f1","ufc","goal","match"},
    "Pets and Animals": {"pet","dog","cat","animals","wildlife","zoo","veterinary","vet","puppy","kitten"},
    "Food and Cooking": {"food","cooking","recipe","kitchen","chef","bake","baking","cook","meal","cuisine"},
    "Science": {"science","physics","chemistry","biology","astronomy","research","lab","experiment","scientific"},
}

YOUTUBE_TO_TARGET_HINTS = {
    "Gaming": "Gaming",
    "Music": "Music",
    "Sports": "Sports",
    "Education": "Education",
    "Pets & Animals": "Pets and Animals",
    "Howto & Style": "Food and Cooking",
    "Science & Technology": "Tech",  # refined with keywords below
}

REGION_CODES = ["US","GB","CA","AU","IN","ZA","DE","FR","ES","IT","BR","MX","TR","SA","AE","EG","ID","MY","SG","PH","JP","KR","VN","TH","NG","PK","NL"]

# ---------- helpers ----------
def now_utc(): return datetime.now(timezone.utc)
def tokens(s): return set(re.findall(r"[a-z0-9]+", (s or "").lower()))
def yt(): return build("youtube","v3",developerKey=API_KEY,cache_discovery=False)

def safe(req):
    try:
        return req.execute()
    except HttpError as e:
        try:
            body = json.loads(e.content.decode("utf-8"))
        except Exception:
            body = {"error": {"message": str(e)}}
        msg = body.get("error", {}).get("message")
        rsn = None
        errs = body.get("error", {}).get("errors", [])
        if errs: rsn = errs[0].get("reason")
        print(f"[HTTP {getattr(e.resp,'status',None)}] reason={rsn} msg={msg}")
        return None

# ---------- search (playlist-free) ----------
def search_channels(ytc, q, region, max_results=10):
    published_after = (now_utc() - timedelta(days=180)).isoformat()
    resp = safe(ytc.search().list(
        part="snippet",
        maxResults=max_results,
        q=q, type="channel",
        regionCode=region,
        relevanceLanguage="en",
        publishedAfter=published_after,
        order="date",  # bias to recent
        safeSearch="none",
    ))
    items = resp.get("items", []) if resp else []
    out = []
    for it in items:
        sn = it.get("snippet", {})
        out.append({
            "channelId": sn.get("channelId"),
            "title": sn.get("title",""),
            "seedQuery": q,
            "region": region,
        })
    return out

def get_recent_video_ids_by_search(ytc, channel_id, limit=12, recent_days=365):
    ids, tok = [], None
    published_after = (now_utc() - timedelta(days=recent_days)).isoformat()
    while len(ids) < limit:
        resp = safe(ytc.search().list(
            part="id",
            channelId=channel_id,
            type="video",
            order="date",
            maxResults=min(50, limit - len(ids)),
            pageToken=tok,
            publishedAfter=published_after,
        ))
        if not resp: break
        for it in resp.get("items", []):
            if it.get("id",{}).get("kind") == "youtube#video":
                ids.append(it["id"]["videoId"])
        tok = resp.get("nextPageToken")
        if not tok: break
    return ids[:limit]

def get_category_map(ytc, region="US"):
    resp = safe(ytc.videoCategories().list(part="snippet", regionCode=region)) or {}
    return {it["id"]: it["snippet"]["title"] for it in resp.get("items", [])}

def fetch_video_snippets(ytc, ids, catmap):
    out = []
    for i in range(0, len(ids), 50):
        resp = safe(ytc.videos().list(part="snippet", id=",".join(ids[i:i+50])))
        if not resp: continue
        for it in resp.get("items", []):
            sn = it.get("snippet", {})
            out.append({
                "title": sn.get("title",""),
                "description": sn.get("description",""),
                "categoryTitle": catmap.get(sn.get("categoryId")),
            })
    return out

# ---------- fixed scoring (per-video cap) ----------
def score_channel_by_recent(ytc, channel_id, target, sample=10):
    ids = get_recent_video_ids_by_search(ytc, channel_id, limit=sample, recent_days=365)
    if not ids: return 0.0
    catmap = get_category_map(ytc, "US")
    metas = fetch_video_snippets(ytc, ids, catmap)
    vk = VALIDATION_KEYWORDS[target]

    direct_hits = 0
    keyword_hits = 0
    for m in metas:
        per_video_direct = 0
        per_video_kw = 0

        ycat = m["categoryTitle"]
        mapped = YOUTUBE_TO_TARGET_HINTS.get(ycat)

        # direct hint (cap at 1 per video)
        if mapped == target:
            per_video_direct = 1
        elif ycat == "Science & Technology":
            t = tokens(m["title"] + " " + m["description"])
            sci = len(t & VALIDATION_KEYWORDS["Science"])
            tech = len(t & VALIDATION_KEYWORDS["Tech"])
            if target == "Science" and sci > tech:
                per_video_direct = 1
            if target == "Tech" and tech >= sci and tech > 0:
                per_video_direct = 1
        elif ycat == "Howto & Style" and target == "Food and Cooking":
            if len(tokens(m["title"] + " " + m["description"]) & VALIDATION_KEYWORDS["Food and Cooking"]) > 0:
                per_video_direct = 1

        # keyword hint (cap at 1 per video)
        if len(tokens(m["title"] + " " + m["description"]) & vk) > 0:
            per_video_kw = 1

        direct_hits += per_video_direct
        keyword_hits += per_video_kw

    n = max(1, len(metas))
    score = 0.7 * (direct_hits / n) + 0.3 * (keyword_hits / n)
    # hard clamp [0,1]
    return max(0.0, min(1.0, score))

# ---------- main pipeline ----------
def main():
    print("PY:", sys.executable)
    print("CWD:", Path.cwd())
    print("__file__:", Path(__file__).resolve())
    print("API key set?:", API_KEY != "YOUR_API_KEY_HERE")
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: set YOUTUBE_API_KEY or hardcode API_KEY"); return

    ytc = yt()
    random.seed()

    out_path = Path.cwd() / "channels_by_category.csv"
    headers = ["channelId","title","category","score","seedQuery","region"]

    # open (append) and write header once
    write_header = not out_path.exists()
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=headers)
    if write_header:
        w.writeheader(); f.flush()
    print(f"[WRITING] -> {out_path}")

    # region round-robin + multi-seed per category
    region_cycle = cycle(random.sample(REGION_CODES, k=len(REGION_CODES)))

    # TUNABLES (start conservative, then raise)
    seeds_per_category = 4      # â†‘ for more breadth
    search_max_results   = 12   # per seed search call
    accept_threshold     = 0.45 # â†‘ for stricter purity
    validate_sample_size = 10   # videos per channel to score
    max_channels_per_cat = 12   # stop after this many accepted per category

    total_written = 0
    for category in TARGETS:
        accepted = 0
        # rotate a different region each seed to ensure diversity
        for q in islice(cycle(random.sample(CATEGORY_SEEDS[category], k=len(CATEGORY_SEEDS[category]))), seeds_per_category):
            region = next(region_cycle)
            candidates = search_channels(ytc, q, region, max_results=search_max_results)
            random.shuffle(candidates)

            for cand in candidates:
                cid = cand["channelId"]
                if not cid: continue
                # score
                s = 0.0
                try:
                    s = score_channel_by_recent(ytc, cid, category, sample=validate_sample_size)
                except Exception as e:
                    print(f"[WARN] score failed {cid}: {e}")
                    s = 0.0

                if s >= accept_threshold:
                    w.writerow({
                        "channelId": cid,
                        "title": cand["title"],
                        "category": category,
                        "score": round(s,3),
                        "seedQuery": cand["seedQuery"],
                        "region": cand["region"],
                    })
                    f.flush()
                    accepted += 1
                    total_written += 1

                if accepted >= max_channels_per_cat:
                    break

        print(f"[{category}] accepted={accepted}")

    f.close()
    print(f"[DONE] wrote {total_written} rows -> {out_path}")

if __name__ == "__main__":
    main()
