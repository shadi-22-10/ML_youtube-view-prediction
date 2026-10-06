import os
import csv
import datetime
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ===================================================
# CONFIG
# ===================================================
API_KEY = "YOUR_API_KEY"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "youtube_dataset.csv")
CHANNEL_FILE = os.path.join(BASE_DIR, "channel_list.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

YOUTUBE = build("youtube", "v3", developerKey=API_KEY)

# ===================================================
# HELPER FUNCTIONS
# ===================================================

def load_csv_dict(filename):
    data = []
    with open(filename, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data, reader.fieldnames if 'reader' in locals() else None

def save_snapshot_csv(data, original_file):
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(original_file).replace(".csv", "")
    snapshot_file = os.path.join(SNAPSHOT_DIR, f"{base_name}_snapshot_{timestamp_str}.csv")

    if not data:
        print(f"No data to save for {original_file}")
        return

    # Read original header to preserve order
    with open(original_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        original_header = next(reader, [])

    # Ensure snapshot_date is included
    if "snapshot_date" not in original_header:
        fieldnames = original_header + ["snapshot_date"]
    else:
        fieldnames = original_header

    normalized_data = []
    for row in data:
        # Remove any None keys
        clean_row = {k: v for k, v in row.items() if k is not None}
        # Fill missing fields
        for fn in fieldnames:
            if fn not in clean_row:
                clean_row[fn] = ""
        # Keep only keys in fieldnames
        clean_row = {k: clean_row[k] for k in fieldnames}
        # Update snapshot date
        clean_row["snapshot_date"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        normalized_data.append(clean_row)

    with open(snapshot_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_data)

    print(f"Snapshot saved: {snapshot_file}")

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def update_channel_stats(channels):
    updated_channels = []
    for chunk in chunk_list(channels, 50):
        ids = [c["channel_id"] for c in chunk]
        try:
            res = YOUTUBE.channels().list(
                part="statistics,snippet",
                id=",".join(ids),
                maxResults=50
            ).execute()
            for ch in res.get("items", []):
                ch_id = ch.get("id")
                for c in chunk:
                    if c["channel_id"] == ch_id:
                        stats = ch.get("statistics", {})
                        snippet = ch.get("snippet", {})
                        c["channel_subscriberCount"] = stats.get("subscriberCount", c.get("channel_subscriberCount",""))
                        c["channel_viewCount"] = stats.get("viewCount", c.get("channel_viewCount",""))
                        c["channel_videoCount"] = stats.get("videoCount", c.get("channel_videoCount",""))
                        c["channel_country"] = snippet.get("country", c.get("channel_country",""))
                        c["channel_publishedAt"] = snippet.get("publishedAt", c.get("channel_publishedAt",""))
                        updated_channels.append(c)
        except HttpError as e:
            print(f"Error updating channels: {e}")
        time.sleep(1)
    return updated_channels

def update_video_stats(videos):
    updated_videos = []
    for chunk in chunk_list(videos, 50):
        ids = [v["video_id"] for v in chunk]
        try:
            res = YOUTUBE.videos().list(
                part="statistics,contentDetails",
                id=",".join(ids),
                maxResults=50
            ).execute()
            for vdata in res.get("items", []):
                vid_id = vdata.get("id")
                for v in chunk:
                    if v["video_id"] == vid_id:
                        stats = vdata.get("statistics", {})
                        cd = vdata.get("contentDetails", {})
                        v["viewCount"] = stats.get("viewCount", v.get("viewCount",""))
                        v["likeCount"] = stats.get("likeCount", v.get("likeCount",""))
                        v["commentCount"] = stats.get("commentCount", v.get("commentCount",""))
                        v["duration"] = cd.get("duration", v.get("duration",""))
                        updated_videos.append(v)
        except HttpError as e:
            print(f"Error updating videos: {e}")
        time.sleep(1)
    return updated_videos

# ===================================================
# MAIN LOGIC
# ===================================================
def main():
    print("Reading existing data...")
    channels, _ = load_csv_dict(CHANNEL_FILE)
    videos, _ = load_csv_dict(DATASET_FILE)

    print(f"Loaded {len(channels)} channels and {len(videos)} videos")

    print("Updating channel stats...")
    updated_channels = update_channel_stats(channels)
    save_snapshot_csv(updated_channels, CHANNEL_FILE)

    print("Updating video stats...")
    updated_videos = update_video_stats(videos)
    save_snapshot_csv(updated_videos, DATASET_FILE)

    print("All snapshots created successfully.")

if __name__ == "__main__":
    main()
