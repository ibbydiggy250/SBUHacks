import os
import json
import requests

# === CONFIG ===
glosses_to_keep = [
     "BUILD"
]
wlasl_file = "WLASL_v0.3.json"
output_folder = r"C:\Users\rumma\OneDrive\Desktop\New folder\SBUHacks\WLASL\start_kit\raw_videos(1)"

# === Ensure folder exists ===
os.makedirs(output_folder, exist_ok=True)

# === Load WLASL metadata ===
with open(wlasl_file, "r", encoding="utf-8") as f:
    wlasl_data = json.load(f)

# === Filter by glosses ===
filtered_entries = [entry for entry in wlasl_data if entry["gloss"].upper() in glosses_to_keep]
print(f"Found {len(filtered_entries)} glosses")

downloaded = 0
skipped = 0

# === Download each matching video ===
for entry in filtered_entries:
    gloss = entry["gloss"].upper()
    gloss_folder = os.path.join(output_folder, gloss)
    os.makedirs(gloss_folder, exist_ok=True)

    for inst in entry["instances"]:
        vid_id = str(inst["video_id"])
        urls = [
            inst.get("url"),
            inst.get("source_url"),
            inst.get("video_url")
        ]
        url = next((u for u in urls if u and "http" in u), None)

        if not url:
            skipped += 1
            print(f"⚠️ No URL found for {gloss} ({vid_id})")
            continue

        dest_path = os.path.join(gloss_folder, f"{vid_id}.mp4")
        if os.path.exists(dest_path):
            print(f"⏩ Already exists: {dest_path}")
            continue

        try:
            print(f"⬇️ Downloading {gloss}: {url}")
            r = requests.get(url, timeout=30, stream=True)
            if r.status_code == 200:
                with open(dest_path, "wb") as f_out:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f_out.write(chunk)
                downloaded += 1
            else:
                print(f"❌ Failed ({r.status_code}): {url}")
                skipped += 1
        except Exception as e:
            print(f"❌ Error downloading {gloss} ({vid_id}): {e}")
            skipped += 1

print(f"\n✅ Downloaded {downloaded} videos to {output_folder}")
print(f"⚠️ Skipped {skipped} (unavailable or failed)")
