#!/usr/bin/env python3
"""Assemble the primary benchmark from downloaded images and publication labels."""

import argparse
import csv
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_image(directory, stem):
    matches = [directory / f"{stem}{suffix}" for suffix in IMAGE_SUFFIXES]
    matches += [directory / f"{stem}{suffix.upper()}" for suffix in IMAGE_SUFFIXES]
    matches = sorted({path for path in matches if path.is_file()})
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one image for {stem} under {directory}; found {len(matches)}")
    return matches[0]


def read_ids(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_image_entry(split, image_name):
    return f"./images/{split}/{image_name}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf100-root", required=True, help="Downloaded RF100 Construction Safety v2 root")
    parser.add_argument("--itppe-root", required=True, help="Downloaded ITPPE archive root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    primary = REPO / "data/primary_benchmark"
    rf_dir = REPO / "data/rf100_v2"
    rf_rows = list(csv.DictReader((rf_dir / "manifest.tsv").open(encoding="utf-8"), delimiter="\t"))
    rf_by_public_id = {Path(row["label_file"]).stem: row for row in rf_rows}
    rf_root = Path(args.rf100_root)
    itppe_root = Path(args.itppe_root)
    output = Path(args.output).resolve()
    itppe_images = itppe_root / "images"
    itppe_labels = itppe_root / "labels"

    for split in ("train", "val", "test"):
        ids = read_ids(primary / f"{split}.txt")
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_names = []
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for public_id in ids:
            if public_id.startswith("rf100_v2_"):
                row = rf_by_public_id[public_id]
                source_image = find_image(rf_root / row["upstream_split"] / "images", row["image_id"])
                source_label = rf_dir / row["label_file"]
            elif public_id.startswith("itppe_"):
                source_image = find_image(itppe_images, public_id)
                source_label = itppe_labels / f"{public_id}.txt"
            else:
                raise ValueError(f"Unknown primary benchmark ID: {public_id}")
            image_name = f"{public_id}{source_image.suffix.lower()}"
            shutil.copy2(source_image, image_dir / image_name)
            shutil.copy2(source_label, label_dir / f"{public_id}.txt")
            image_names.append(image_name)
        (output / f"{split}.txt").write_text(
            "\n".join(split_image_entry(split, name) for name in sorted(image_names)) + "\n",
            encoding="utf-8",
        )

    (output / "dataset.yaml").write_text(
        "path: " + output.resolve().as_posix() + "\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "test: test.txt\n"
        "names:\n"
        "  0: person\n"
        "  1: helmet\n"
        "  2: vest\n",
        encoding="utf-8",
    )
    print("primary_dataset=READY")


if __name__ == "__main__":
    main()
