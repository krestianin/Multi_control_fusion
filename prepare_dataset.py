import csv
import json
from pathlib import Path

# Change this to your DiffusionDB root folder
ROOT = Path(r"data")
OUT_CSV = "train.csv"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def find_prompt_in_obj(obj):
    """
    Recursively search for a likely prompt field inside JSON data.
    """
    if isinstance(obj, dict):
        # Try common field names first
        for key in ("p", "prompt", "caption", "text"):
            if key in obj and isinstance(obj[key], str) and obj[key].strip():
                return obj[key].strip()

        # Recurse into nested dict values
        for value in obj.values():
            result = find_prompt_in_obj(value)
            if result:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = find_prompt_in_obj(item)
            if result:
                return result

    return None


def build_image_prompt_map_from_json(json_path):
    """
    Build a map:
        image filename stem -> prompt
    from one JSON file.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read JSON: {json_path} -> {e}")
        return {}

    mapping = {}

    # Case 1: dict keyed by image names / ids
    if isinstance(data, dict):
        for key, value in data.items():
            prompt = find_prompt_in_obj(value)
            if prompt:
                mapping[Path(str(key)).stem] = prompt

    # Case 2: list of objects, maybe each has filename/id + prompt
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            prompt = find_prompt_in_obj(item)
            if not prompt:
                continue

            filename = None
            for k in ("file_name", "filename", "image", "image_path", "path", "id"):
                if k in item:
                    filename = str(item[k])
                    break

            if filename:
                mapping[Path(filename).stem] = prompt

    return mapping


def main():
    all_json_files = list(ROOT.rglob("*.json"))
    all_image_files = [p for p in ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

    if not all_json_files:
        print("[ERROR] No JSON files found.")
        return

    if not all_image_files:
        print("[ERROR] No image files found.")
        return

    print(f"Found {len(all_json_files)} JSON files")
    print(f"Found {len(all_image_files)} image files")

    # Build one big stem -> prompt map
    prompt_map = {}
    for json_file in all_json_files:
        local_map = build_image_prompt_map_from_json(json_file)
        prompt_map.update(local_map)

    written = 0
    skipped = 0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "caption"])

        for img_path in sorted(all_image_files):
            stem = img_path.stem
            prompt = prompt_map.get(stem)

            if not prompt:
                skipped += 1
                continue

            writer.writerow([str(img_path.resolve()), prompt])
            written += 1

    print(f"[DONE] Wrote: {OUT_CSV}")
    print(f"[DONE] Rows written: {written}")
    print(f"[DONE] Images skipped (no prompt found): {skipped}")


if __name__ == "__main__":
    main()