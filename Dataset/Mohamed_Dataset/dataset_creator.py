import os
import csv
import datetime
import time
import requests
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json

# ===================================================
# CONFIG
# ===================================================
API_KEY = "API_KEY_HERE" 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHANNEL_LIST_FILE = os.path.join(BASE_DIR, "channel_list.csv")
DATASET_FILE = os.path.join(BASE_DIR, "youtube_dataset.csv")
THUMBNAIL_DIR = os.path.join(BASE_DIR, "thumbnails")
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

YOUTUBE = build("youtube", "v3", developerKey=API_KEY)

# Simple API call counters for diagnostics
API_CALL_COUNT = {
    "search": 0,
    "channels": 0,
    "playlistItems": 0,
    "videos": 0,
}

# Flag to indicate we've hit quota and should stop
QUOTA_EXCEEDED = False

# Set a soft API budget (set to None to disable budgeting)
# Default: None (disabled). Set a number to enable a soft stop before quota is exhausted.
BUDGET = None

# Conservative per-endpoint weights (tune as needed). Edit these in-code to change behavior.
WEIGHTS = {
    "search": 5,        # search calls are relatively cheap but numerous
    "channels": 50,     # channels().list can be somewhat costly
    "playlistItems": 2, # playlistItems are cheap
    "videos": 100       # videos().list with statistics/contentDetails is expensive
}

# If True, fetch full statistics (viewCount/likeCount/commentCount etc.) for all collected videos.
# This is expensive. Keep False to only fetch full stats for the top K per channel (safer for quota).
FETCH_STATS_FOR_ALL = True

def consume_budget(endpoint):
    """Decrement the global budget by the weight for endpoint. Returns True if still allowed, False if budget exhausted."""
    global BUDGET, QUOTA_EXCEEDED
    if BUDGET is None:
        return True
    weight = WEIGHTS.get(endpoint, 10)
    BUDGET -= weight
    API_CALL_COUNT[endpoint] = API_CALL_COUNT.get(endpoint, 0) + 1
    if BUDGET <= 0:
        QUOTA_EXCEEDED = True
        print(f"Soft budget exhausted after consuming {weight} for {endpoint}. Remaining budget: {BUDGET}")
        return False
    return True

# Simple persistent cache file for channel stats
CACHE_FILE = os.path.join(BASE_DIR, "channel_stats_cache.json")


def load_cache(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def periodic_diagnostic_print(counter, every=10):
    # simple print helper to show counters
    total = sum(counter.values())
    if total % every == 0:
        print("API_CALL_COUNT:", counter)

# Filters
MIN_SUBS = 100_000
MIN_VIDEOS = 50
MAX_CHANNELS = 50
VIDEOS_PER_CHANNEL = 50
# Limit how many search pages we will scan per keyword to avoid unbounded quota use
MAX_SEARCH_PAGES = 1
# Limit how many search pages we will scan per channel when listing videos
MAX_VIDEO_SEARCH_PAGES = 4
# When collecting videos per channel, fetch full statistics for the top K most-recent videos only
VIDEOS_STATS_TOP_K = 10
# Hard cap on how many video IDs to collect per channel to avoid runaway
MAX_VIDS_PER_CHANNEL = 200

# Date range: last 1 year
today = datetime.datetime.now(datetime.timezone.utc)
# include videos published within the past 365 days up to now
max_date = today.isoformat()
min_date = (today - datetime.timedelta(days=365)).isoformat()

# Hardcoded 30 categories
CATEGORIES = [
    "Food", "Tech", "Travel", "Music", "Sports", "Gaming", "Education",
    "News", "Comedy", "Health", "Fitness", "Beauty", "Fashion", "DIY",
    "Movies", "Art", "Photography", "Science", "History", "Business",
    "Cars", "Animals", "Nature", "Animation", "Lifestyle", "Books",
    "Technology Reviews", "Cooking", "Vlogs", "Finance"
]

# Best-effort mapping from our human category names to YouTube videoCategoryId values.
# Edit these mappings as needed. Adding videoCategoryId to the search request filters
# results to that YouTube category and does not increase API call count.
CATEGORY_TO_VIDEO_ID = {
    "Music": "10",
    "Gaming": "20",
    "Sports": "17",
    "Travel": "19",
    "Education": "27",
    "Comedy": "23",
    "Tech": "28",
    "Technology Reviews": "28",
    "Fashion": "26",
    "Cooking": "26",
    "Food": "26",  # best-effort: map Food/Cooking to Howto & Style
    "Vlogs": "21",
    "News": "25",
    "Movies": "30",
    "Animation": "31",
    "Science": "28",
    "Photography": "27",
    "Business": "25",
    "Finance": "25",
}

# ===================================================
# HELPER FUNCTIONS
# ===================================================
def save_csv(filename, rows, header=None):
    # Ensure header alignment: if file doesn't exist, write header; if file exists but header provided,
    # verify header matches number of columns in rows and append missing header fields if needed.
    file_exists = os.path.isfile(filename)
    if header is not None:
        # normalize header to list
        header = list(header)
    with open(filename, "a+", newline="", encoding="utf-8") as f:
        f.seek(0)
        existing_header = None
        try:
            reader = csv.reader(f)
            existing_rows = list(reader)
            if existing_rows:
                existing_header = existing_rows[0]
        except Exception:
            existing_header = None

        writer = csv.writer(f)
        if not file_exists and header:
            writer.writerow(header)
        elif file_exists and header and existing_header is None:
            # file exists but empty or malformed, write header
            writer.writerow(header)
        # If rows are provided, ensure each row length matches header length by padding with empty strings
        if header:
            expected = len(header)
            padded = []
            for r in rows:
                if len(r) < expected:
                    padded.append(list(r) + [""] * (expected - len(r)))
                else:
                    padded.append(r)
            writer.writerows(padded)
        else:
            writer.writerows(rows)


def load_existing_channels():
    existing = set()
    if os.path.exists(CHANNEL_LIST_FILE):
        with open(CHANNEL_LIST_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "channel_id" in row:
                    existing.add(row["channel_id"])
    return existing


def load_existing_videos():
    existing = set()
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "video_id" in row:
                    existing.add(row["video_id"])
    return existing


def download_thumbnail(url, filename, retries=3, timeout=10):
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)
            return True
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt} failed: {e}")
            time.sleep(2)
    print(f"Failed to download thumbnail {filename} after {retries} attempts.")
    return False


def get_next_category_from_list():
    used_categories = set()
    if os.path.exists(CHANNEL_LIST_FILE):
        with open(CHANNEL_LIST_FILE, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat = row.get("category", "").strip()
                if cat:
                    used_categories.add(cat)
    for c in CATEGORIES:
        if c not in used_categories:
            return c
    return None


def extract_category_from_row(row, fallback=None):
    """Return the category value from a channel-list CSV row using common header names.
    Falls back to `fallback` if none found or empty.
    """
    for key in ("category", "Category", "category_name", "cat", "genre"):
        val = row.get(key)
        if val:
            return val
    return fallback


def fetch_channels_by_keyword(keyword, max_pages=1, start_page_token=None):
    # Use video search to discover channel IDs, then batch channels().list for stats.
    # This function performs up to `max_pages` search pages starting from start_page_token
    # and returns (channels, next_page_token). It consults the persistent cache.
    channels = []
    collected_ids = set()
    next_page_token = start_page_token
    pages = 0
    # Avoid re-checking channels already present in the channel list file
    existing_channel_ids = load_existing_channels()
    # load persistent cache
    cache = load_cache(CACHE_FILE)
    try:
        while len(channels) < MAX_CHANNELS and pages < max_pages:
            # budget-aware search call
            if not consume_budget("search"):
                break
            # If we have a mapping to an official YouTube videoCategoryId, include it to reduce noise
            vid_cat = CATEGORY_TO_VIDEO_ID.get(keyword)
            req_kwargs = dict(
                q=keyword,
                type="video",
                part="snippet",
                maxResults=50,
                pageToken=next_page_token,
                publishedAfter=min_date,
                publishedBefore=max_date,
                order="date"
            )
            if vid_cat:
                req_kwargs["videoCategoryId"] = vid_cat
            req = YOUTUBE.search().list(**req_kwargs)
            res = req.execute()
            items = res.get("items", [])

            # collect unique channel IDs from this page, skipping ones we already have
            channel_ids = []
            for item in items:
                ch_id = item.get("snippet", {}).get("channelId")
                if not ch_id:
                    continue
                if ch_id in collected_ids or ch_id in existing_channel_ids:
                    continue
                channel_ids.append(ch_id)
                collected_ids.add(ch_id)

            # batch channel stats requests
            for i in range(0, len(channel_ids), 50):
                chunk = channel_ids[i:i+50]
                if not chunk:
                    continue
                # First, consult cache entries for these ids
                to_fetch = []
                for cid in chunk:
                    if cid in cache:
                        entry = cache[cid]
                        subs = int(entry.get("subscriberCount", 0))
                        vids = int(entry.get("videoCount", 0))
                        if subs >= MIN_SUBS and vids >= MIN_VIDEOS and cid not in existing_channel_ids:
                            if cid not in [c["channel_id"] for c in channels]:
                                channels.append({
                                    "channel_name": entry.get("title", ""),
                                    "channel_id": cid,
                                    "channel_subscriberCount": subs,
                                    "channel_viewCount": entry.get("viewCount", ""),
                                    "channel_videoCount": vids,
                                    "channel_country": entry.get("country", ""),
                                    "channel_publishedAt": entry.get("publishedAt", ""),
                                    "category": keyword,
                                    "snapshot_date": datetime.datetime.now(datetime.timezone.utc).isoformat()
                                })
                                if len(channels) >= MAX_CHANNELS:
                                    break
                    else:
                        to_fetch.append(cid)
                if len(channels) >= MAX_CHANNELS:
                    break

                if to_fetch:
                    if not consume_budget("channels"):
                        break
                    ch_req = YOUTUBE.channels().list(
                        id=','.join(to_fetch),
                        part="statistics,snippet",
                        maxResults=50
                    )
                    ch_res = ch_req.execute()
                    for ch in ch_res.get("items", []):
                        cid = ch.get("id")
                        stats = ch.get("statistics", {})
                        subs = int(stats.get("subscriberCount", 0))
                        vids = int(stats.get("videoCount", 0))
                        # populate cache
                        cache[cid] = {
                            "title": ch.get("snippet", {}).get("title", ""),
                            "subscriberCount": stats.get("subscriberCount", 0),
                            "videoCount": stats.get("videoCount", 0),
                            "viewCount": stats.get("viewCount", ""),
                            "country": ch.get("snippet", {}).get("country", ""),
                            "publishedAt": ch.get("snippet", {}).get("publishedAt", "")
                        }
                        if subs >= MIN_SUBS and vids >= MIN_VIDEOS and cid not in existing_channel_ids:
                            if cid not in [c["channel_id"] for c in channels]:
                                channels.append({
                                    "channel_name": ch["snippet"]["title"],
                                    "channel_id": cid,
                                    "channel_subscriberCount": subs,
                                    "channel_viewCount": stats.get("viewCount", ""),
                                    "channel_videoCount": vids,
                                    "channel_country": ch["snippet"].get("country", ""),
                                    "channel_publishedAt": ch["snippet"]["publishedAt"],
                                    "category": keyword,
                                    "snapshot_date": datetime.datetime.now(datetime.timezone.utc).isoformat()
                                })
                                if len(channels) >= MAX_CHANNELS:
                                    break
                # end batch handling
                if len(channels) >= MAX_CHANNELS:
                    break

            next_page_token = res.get("nextPageToken")
            pages += 1
            if not next_page_token:
                break
            time.sleep(1)
    except HttpError as e:
        print("Error fetching channels:", e)
        print("API call counts:", API_CALL_COUNT)
    # persist cache
    try:
        save_cache(CACHE_FILE, cache)
    except Exception:
        pass
    print(f"Fetched {len(channels)} channels for keyword '{keyword}'")
    return channels


def fetch_videos_from_channel(channel_id, existing_videos, category_name):
    videos = []
    global QUOTA_EXCEEDED
    try:
        # Get uploads playlist for the channel (budget-aware)
        if not consume_budget("channels"):
            return videos
        ch_res = YOUTUBE.channels().list(id=channel_id, part="contentDetails").execute()
        uploads = None
        for item in ch_res.get("items", []):
            uploads = item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
            break
        if not uploads:
            return videos

        # Page through the uploads playlist to collect snippet info and video IDs (cheaper)
        collected_vid_ids = []
        snippet_map = {}  # vid_id -> snippet fields
        next_token = None
        pages = 0
        while pages < MAX_VIDEO_SEARCH_PAGES and len(collected_vid_ids) < MAX_VIDS_PER_CHANNEL:
            if not consume_budget("playlistItems"):
                break
            res = YOUTUBE.playlistItems().list(playlistId=uploads, part="snippet,contentDetails", maxResults=50, pageToken=next_token).execute()
            for it in res.get("items", []):
                vid = it.get("contentDetails", {}).get("videoId")
                sn = it.get("snippet", {})
                if not vid:
                    continue
                if vid in existing_videos:
                    continue
                if vid in collected_vid_ids:
                    continue
                collected_vid_ids.append(vid)
                # store snippet-level data (title, description, thumbnails, publishedAt, channelTitle, channelId, tags if present)
                snippet_map[vid] = {
                    "channelTitle": sn.get("channelTitle", ""),
                    "channelId": sn.get("channelId", ""),
                    "title": sn.get("title", ""),
                    "description": sn.get("description", ""),
                    "publishedAt": sn.get("publishedAt", ""),
                    "thumbnails": sn.get("thumbnails", {}),
                    "tags": sn.get("tags", [])
                }
                if len(collected_vid_ids) >= MAX_VIDS_PER_CHANNEL:
                    break
            next_token = res.get("nextPageToken")
            pages += 1
            if not next_token:
                break

        # Fetch full statistics for all newly collected videos (so view/like/comment are populated)
        # Note: this is more expensive in quota. Set BUDGET or WEIGHTS accordingly if you hit limits.
        stats_ids = collected_vid_ids
        stats_map = {}
        for i in range(0, len(stats_ids), 50):
            chunk = stats_ids[i:i+50]
            if not chunk:
                continue
            if not consume_budget("videos"):
                break
            try:
                vid_res = YOUTUBE.videos().list(part="statistics,contentDetails,liveStreamingDetails", id=','.join(chunk)).execute()
            except HttpError as e:
                if "quotaExceeded" in str(e):
                    QUOTA_EXCEEDED = True
                    print(f"Quota exceeded while fetching video stats for channel {channel_id}: {e}")
                    break
                else:
                    print(f"Error fetching video stats: {e}")
                    continue
            for v in vid_res.get("items", []):
                vid_id = v.get("id")
                stats_map[vid_id] = {
                    "viewCount": v.get("statistics", {}).get("viewCount", ""),
                    "likeCount": v.get("statistics", {}).get("likeCount", ""),
                    "commentCount": v.get("statistics", {}).get("commentCount", ""),
                    "contentDetails": v.get("contentDetails", {}),
                    "liveStreamingDetails": v.get("liveStreamingDetails", {})
                }

        # Build rows: fill snippet fields, and use stats_map for numeric fields where available
        for vid_id in collected_vid_ids:
            sn = snippet_map.get(vid_id, {})
            st = stats_map.get(vid_id, {})
            cd = st.get("contentDetails", {}) if st else {}
            ld = st.get("liveStreamingDetails", {}) if st else {}

            thumbs = sn.get("thumbnails", {}) if sn else {}
            thumb = None
            if thumbs:
                thumb = (thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}).get("url")

            thumbnail_file = f"{vid_id}.jpg"
            thumbnail_path = os.path.join(THUMBNAIL_DIR, thumbnail_file)
            if thumb:
                download_thumbnail(thumb, thumbnail_path)

                videos.append([
                vid_id,
                sn.get("channelTitle", ""),
                category_name,
                sn.get("channelId", ""),
                st.get("viewCount", ""),
                st.get("likeCount", ""),
                st.get("commentCount", ""),
                sn.get("title", ""),
                "",  # description intentionally left empty to avoid long text
                ",".join(sn.get("tags", [])) if sn and "tags" in sn else "",
                sn.get("publishedAt", ""),
                cd.get("duration", ""),
                cd.get("dimension", ""),
                cd.get("definition", ""),
                cd.get("caption", ""),
                cd.get("licensedContent", ""),
                cd.get("projection", ""),
                ld.get("liveActualStartTime", ""),
                ld.get("liveScheduledStartTime", ""),
                ld.get("concurrentViewers", ""),
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                thumbnail_file
            ])
            existing_videos.add(vid_id)
    except HttpError as e:
        # set quota flag and stop further work if we've exceeded quota
        if "quotaExceeded" in str(e):
            QUOTA_EXCEEDED = True
            print(f"Quota exceeded while fetching videos for {channel_id}: {e}")
        else:
            print(f"Error fetching videos for {channel_id}: {e}")
        print("API call counts:", API_CALL_COUNT)
    return videos


def ensure_column_in_csv(filename, column_name, default=""):
    """Ensure the CSV at filename has column_name in its header.
    If the file exists and the header doesn't contain the column, rewrite the file
    adding the column at the end and padding existing rows with the default value.
    """
    if not os.path.exists(filename):
        return
    with open(filename, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        # empty file - nothing to do
        return
    header = rows[0]
    if column_name in header:
        return
    header.append(column_name)
    new_rows = [header]
    for row in rows[1:]:
        # pad row to header length
        if len(row) < len(header):
            row = list(row) + [default] * (len(header) - len(row))
        new_rows.append(row)
    # rewrite file
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)


# ===================================================
# MAIN LOGIC
# ===================================================
def main():
    existing_channels = load_existing_channels()
    existing_videos = load_existing_videos()
    print(f"Existing channels: {len(existing_channels)}, existing videos: {len(existing_videos)}")

    # Run-time seen set to avoid adding duplicates during this run
    seen_video_ids = set(existing_videos)

    # load persistent cache (already used by fetch_channels_by_keyword)
    cache = load_cache(CACHE_FILE)

    next_category = get_next_category_from_list()
    if not next_category:
        print("All categories have already been processed. Exiting.")
        return
    print(f"Fetching data for new category: {next_category}")

    # FETCH CHANNELS in 1-page batches and process their videos; only fetch more pages if quota allows
    collected_all = []
    page_token = None
    while not QUOTA_EXCEEDED and len(collected_all) < MAX_CHANNELS:
        collected = fetch_channels_by_keyword(next_category, max_pages=1, start_page_token=page_token)
        # fetch_channels_by_keyword currently returns a list; we can't get next_page_token directly
        # so rely on file-based channel_list.csv growth: write collected, then process those channels
        collected = [c for c in collected if c["channel_id"] not in existing_channels]
        if collected:
            # ensure snapshot_date column exists in channel list
            ensure_column_in_csv(CHANNEL_LIST_FILE, "snapshot_date")
            save_csv(CHANNEL_LIST_FILE, [[c[k] for k in c] for c in collected], header=list(collected[0].keys()))
            existing_channels.update([c["channel_id"] for c in collected])
            collected_all.extend(collected)
            print(f"Saved {len(collected)} new channels for category '{next_category}'.")
        else:
            # no new channels in this page, stop
            break
        # If quota isn't exceeded and we still need more channels, continue looping
        if QUOTA_EXCEEDED:
            break

    # FETCH VIDEOS for channels we just collected (or existing ones if already present)
    with open(CHANNEL_LIST_FILE, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        for row in tqdm(reader, desc=f"Videos ({next_category})"):
            if QUOTA_EXCEEDED:
                print("Quota exceeded — stopping video fetch loop.")
                break
            channel_id = row["channel_id"]
            # diagnostic print
            periodic_diagnostic_print(API_CALL_COUNT, every=25)
            category_name = extract_category_from_row(row, fallback=next_category)
            videos = fetch_videos_from_channel(channel_id, existing_videos, category_name)
            if videos:
                # ensure snapshot_date column exists in dataset
                ensure_column_in_csv(DATASET_FILE, "snapshot_date")
                # Filter out any duplicates discovered during this run
                new_videos = [v for v in videos if v[0] not in seen_video_ids]
                if new_videos:
                    save_csv(
                        DATASET_FILE,
                        new_videos,
                        header=[
                            "video_id", "channel_name", "category", "channel_id", "viewCount",
                            "likeCount", "commentCount", "title", "description", "tags", "publishedAt",
                            "duration", "dimension", "definition", "caption", "licensedContent",
                            "projection", "liveActualStartTime", "liveScheduledStartTime",
                            "concurrentViewers", "snapshot_date", "thumbnail_file"
                        ]
                    )
                    # mark them as seen and update the existing_videos set
                    seen_video_ids.update([v[0] for v in new_videos])
                    existing_videos.update([v[0] for v in new_videos])

    print("Done — dataset built successfully.")


if __name__ == "__main__":
    # Config knobs are set at the top of this file; edit them directly if you want different behavior.
    print("Starting run with config:")
    print(f"  BUDGET={BUDGET}")
    print(f"  FETCH_STATS_FOR_ALL={FETCH_STATS_FOR_ALL}")
    print(f"  MAX_VIDEO_SEARCH_PAGES={MAX_VIDEO_SEARCH_PAGES}")
    print(f"  VIDEOS_STATS_TOP_K={VIDEOS_STATS_TOP_K}")
    print(f"  MAX_VIDS_PER_CHANNEL={MAX_VIDS_PER_CHANNEL}")
    print(f"  WEIGHTS={WEIGHTS}")
    main()
