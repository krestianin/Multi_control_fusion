# using real-picture dataset from Flickr30k and AI-generated dataset from DiffusionDB to create a combined CSV for training

import csv
import json
from pathlib import Path

IMAGES_DIR  = Path("flickr30k-images")
TOKEN_FILE  = Path("flickr30k-descriptions/results_20130124.token")
OUT_CSV         = "train.csv"
OUT_CSV_FLICKR  = "train_flickr.csv"
OUT_CSV_AI      = "train_ai.csv"
CAPTIONS_TO_USE = 4  # concatenate first N captions (out of 5) into one row per image
MAX_IMAGES = 10_000  # flickr images limit

AI_ROOT = Path("data")  # AI-generated dataset root (DiffusionDB)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def build_ai_prompt_map(root: Path) -> dict[str, str]:
    prompt_map: dict[str, str] = {}
    for json_path in root.rglob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read JSON: {json_path} -> {e}")
            continue
        items = data.items() if isinstance(data, dict) else enumerate(data)
        for key, value in items:
            prompt = None
            obj = value if isinstance(value, dict) else {}
            for field in ("p", "prompt", "caption", "text"):
                if field in obj and isinstance(obj[field], str) and obj[field].strip():
                    prompt = obj[field].strip()
                    break
            if prompt:
                prompt_map[Path(str(key)).stem] = prompt
    return prompt_map


def main():
    # --- Flickr dataset ---
    captions: dict[str, list[tuple[int, str]]] = {}
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, caption = line.split("\t", 1)
            filename, idx = key.rsplit("#", 1)
            captions.setdefault(filename, []).append((int(idx), caption))

    flickr_rows: list[tuple[str, str]] = []
    skipped = 0
    for filename, cap_list in sorted(captions.items()):
        if len(flickr_rows) >= MAX_IMAGES:
            break
        img_path = IMAGES_DIR / filename
        if not img_path.exists():
            skipped += 1
            continue
        cap_list.sort(key=lambda x: x[0])
        caption = " ".join(c for _, c in cap_list[:CAPTIONS_TO_USE])
        flickr_rows.append((str(img_path.resolve()), caption))

    print(f"[Flickr] {len(flickr_rows)} rows ({skipped} skipped)")

    # --- AI-generated dataset ---
    ai_rows: list[tuple[str, str]] = []
    if AI_ROOT.exists():
        prompt_map = build_ai_prompt_map(AI_ROOT)
        ai_images = [p for p in AI_ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        for img_path in sorted(ai_images):
            prompt = prompt_map.get(img_path.stem)
            if not prompt:
                continue
            ai_rows.append((str(img_path.resolve()), prompt))
    else:
        print(f"[WARN] AI dataset root not found: {AI_ROOT}")

    print(f"[AI]    {len(ai_rows)} rows")

    def write_csv(path: str, rows: list[tuple[str, str]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "caption"])
            writer.writerows(rows)

    write_csv(OUT_CSV_FLICKR, flickr_rows)
    write_csv(OUT_CSV_AI, ai_rows)
    write_csv(OUT_CSV, flickr_rows + ai_rows)
    print(f"[DONE]  {len(flickr_rows) + len(ai_rows)} rows total → {OUT_CSV}, {OUT_CSV_FLICKR}, {OUT_CSV_AI}")


if __name__ == "__main__":
    main()