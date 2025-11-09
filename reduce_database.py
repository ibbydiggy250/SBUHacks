import json, random

# === CONFIG ===
input_file = "asl_database_gloss.json"     # your full database
output_file = "asl_database_small.json"    # new trimmed file
max_signs = 100                            # how many signs to keep
max_samples_per_sign = 25                  # how many samples per sign (to balance size)

# Load full database
print(f"Loading {input_file} ...")
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Loaded {len(data)} signs")

# --- Pick the most frequent signs ---
# Sort signs by sample count (descending)
sorted_signs = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)

# Keep only the top N
subset = sorted_signs[:max_signs]

# --- Trim each sign to a limited number of samples ---
new_data = {}
for sign, samples in subset:
    if len(samples) > max_samples_per_sign:
        samples = random.sample(samples, max_samples_per_sign)
    new_data[sign] = samples

print(f"Kept {len(new_data)} signs "
      f"({sum(len(v) for v in new_data.values())} total samples)")

# --- Save ---
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2)

print(f"✅ Saved smaller database to {output_file}")
