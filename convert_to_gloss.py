import json, os

print("=== Script started ===")

import os
print("Current working directory:", os.getcwd())

gloss_map_path = r"C:\Users\rumman\OneDrive\Desktop\New folder\SBUHacks\WLASL_v0.3.json"
print("Checking mapping file:", os.path.exists(gloss_map_path))

# --- Load WLASL mapping ---
gloss_map_path = r"C:\Users\rumma\OneDrive\Desktop\New folder\SBUHacks\WLASL_v0.3.json"
with open(gloss_map_path, "r", encoding="utf-8") as f:
    entries = json.load(f)

video_to_gloss = {}
for entry in entries:
    gloss = entry["gloss"].upper()
    for instance in entry["instances"]:
        video_to_gloss[str(instance["video_id"])] = gloss

print(f"Loaded {len(video_to_gloss)} video-to-gloss mappings")

# --- Load your current database ---
with open("hack_database.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries from hack_database.json")

# --- Convert keys from video IDs to gloss names ---
new_data = {}
for key, samples in data.items():
    base = os.path.splitext(os.path.basename(key))[0]  # e.g. 69547
    gloss = video_to_gloss.get(base, base)  # default to filename if not found
    new_data.setdefault(gloss, []).extend(samples)

print(f"Converted {len(new_data)} glosses")


# --- Save the new database ---
out_path = "asl_database_gloss.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2)

print(f"✅ Converted database saved as {out_path}")
