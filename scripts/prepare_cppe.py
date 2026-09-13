#!/usr/bin/env python3
"""Convert CPPE Pascal VOC annotations using the frozen publication membership."""

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CPPE = REPO / "data/cppe"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_SPLIT_COUNTS = {"train": 746, "val": 93, "test": 93}


def read_mapping(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    mapping = {row["source_class"].strip().lower(): (int(row["target_id"]), row["target_class"]) for row in rows}
    if mapping != {"worker": (0, "person"), "hardhat": (1, "helmet"), "vest": (2, "vest")}:
        raise ValueError(f"Unexpected CPPE mapping: {mapping}")
    return mapping


def read_memberships():
    memberships = {}
    all_ids = []
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        ids = [line.strip() for line in (CPPE / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f"{split} membership must contain {expected} unique IDs; found {len(ids)}")
        if any(not public_id.startswith("cppe_") for public_id in ids):
            raise ValueError(f"Unexpected public ID in {split}.txt")
        memberships[split] = ids
        all_ids.extend(ids)
    if len(set(all_ids)) != sum(EXPECTED_SPLIT_COUNTS.values()):
        raise ValueError("CPPE train/val/test memberships overlap")
    return memberships


def index_images(directory):
    images = {}
    duplicate_stems = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in images:
            duplicate_stems.append(path.stem)
        else:
            images[path.stem] = path
    if duplicate_stems:
        raise ValueError(f"Duplicate source image stems: {sorted(set(duplicate_stems))[:20]}")
    return images


def required_float(node, name, public_id):
    value = node.findtext(name)
    if value is None:
        raise ValueError(f"Missing {name} for {public_id}")
    return float(value)


def convert_annotation(xml_path, public_id, mapping):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size for {public_id}")
    width = required_float(size, "width", public_id)
    height = required_float(size, "height", public_id)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for {public_id}: {width}x{height}")

    labels = []
    counts = Counter()
    for obj in root.findall("object"):
        source_class = (obj.findtext("name") or "").strip().lower()
        if source_class not in mapping:
            raise ValueError(f"Unmapped CPPE class {source_class!r} in {public_id}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox for {public_id}/{source_class}")
        xmin = min(width, max(0.0, required_float(box, "xmin", public_id)))
        ymin = min(height, max(0.0, required_float(box, "ymin", public_id)))
        xmax = min(width, max(0.0, required_float(box, "xmax", public_id)))
        ymax = min(height, max(0.0, required_float(box, "ymax", public_id)))
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid box for {public_id}/{source_class}: {(xmin, ymin, xmax, ymax)}")

        target_id, target_class = mapping[source_class]
        xc = ((xmin + xmax) / 2.0) / width
        yc = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        labels.append(f"{target_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
        counts[target_class] += 1
    return labels, counts


def audit_source(source, memberships, mapping):
    image_dir = source / "JPEGImages"
    annotation_dir = source / "Annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError("CPPE source must contain JPEGImages/ and Annotations/")

    images = index_images(image_dir)
    xmls = {path.stem: path for path in sorted(annotation_dir.glob("*.xml"))}
    expected_stems = {public_id.removeprefix("cppe_") for ids in memberships.values() for public_id in ids}
    missing_images = sorted(expected_stems - set(images))
    missing_xml = sorted(expected_stems - set(xmls))
    extra_images = sorted(set(images) - expected_stems)
    extra_xml = sorted(set(xmls) - expected_stems)
    print(f"source_images={len(images)} source_xml={len(xmls)}")
    print(f"missing_images={len(missing_images)} missing_xml={len(missing_xml)}")
    print(f"extra_images={len(extra_images)} extra_xml={len(extra_xml)}")
    if missing_images or missing_xml or extra_images or extra_xml:
        raise ValueError(
            "CPPE source/membership mismatch: "
            f"missing_images={missing_images[:10]} missing_xml={missing_xml[:10]} "
            f"extra_images={extra_images[:10]} extra_xml={extra_xml[:10]}"
        )

    records = {}
    total_counts = Counter()
    for split, ids in memberships.items():
        for public_id in ids:
            stem = public_id.removeprefix("cppe_")
            labels, counts = convert_annotation(xmls[stem], public_id, mapping)
            records[public_id] = (images[stem], labels)
            total_counts.update(counts)
        print(f"{split}_images={len(ids)}")
    for target_class in ("person", "helmet", "vest"):
        print(f"{target_class}_instances={total_counts[target_class]}")
    return records


def write_dataset(output, memberships, records):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    for split, ids in memberships.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        list_lines = []
        for public_id in ids:
            source_image, labels = records[public_id]
            image_name = f"{public_id}{source_image.suffix.lower()}"
            shutil.copy2(source_image, image_dir / image_name)
            (label_dir / f"{public_id}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
            )
            list_lines.append(f"./images/{split}/{image_name}")
        (output / f"{split}.txt").write_text("\n".join(list_lines) + "\n", encoding="utf-8")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="CPPE root containing JPEGImages and Annotations")
    parser.add_argument("--output", help="New YOLO dataset directory")
    parser.add_argument("--validate-only", action="store_true", help="Audit conversion without writing files")
    args = parser.parse_args()
    if not args.validate_only and not args.output:
        parser.error("--output is required unless --validate-only is used")

    mapping = read_mapping(CPPE / "mapping.tsv")
    memberships = read_memberships()
    records = audit_source(Path(args.source), memberships, mapping)
    if args.validate_only:
        print("cppe_conversion=VALIDATED_NO_WRITE")
        return
    write_dataset(Path(args.output).resolve(), memberships, records)
    print("cppe_conversion=PASS")


if __name__ == "__main__":
    main()
