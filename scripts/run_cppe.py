#!/usr/bin/env python3
"""Run CPPE Protocol A or B from its publication configuration."""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[1]
CPPE = REPO / "data/cppe"


def load_protocol(name):
    path = CPPE / f"protocol_{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def data_path(config, override):
    configured = config.pop("data")
    if override:
        return Path(override)
    path = Path(configured)
    return path if path.is_absolute() else CPPE / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("a", "b"), required=True)
    parser.add_argument("--model", required=True, help="Primary checkpoint for A; model YAML for B")
    parser.add_argument("--data", help="Optional prepared CPPE dataset YAML override")
    parser.add_argument("--project", default="runs/cppe")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    config = load_protocol(args.protocol)
    dataset = data_path(config, args.data)
    config.pop("protocol")

    if args.protocol == "a":
        config["split"] = config.pop("endpoint")
        config["batch"] = config.pop("eval_batch")
        YOLO(args.model).val(data=str(dataset), device=args.device, **config)
        return

    train_batch = config.pop("train_batch")
    eval_batch = config.pop("eval_batch")
    selection_split = config.pop("selection_split")
    final_endpoint = config.pop("final_endpoint")

    eval_half = config.pop("eval_half")
    eval_conf = config.pop("eval_conf")
    eval_iou = config.pop("eval_iou")
    eval_max_det = config.pop("eval_max_det")
    eval_augment = config.pop("eval_augment")

    workers = config["workers"]
    imgsz = config["imgsz"]
    model = YOLO(args.model)
    model.train(
        data=str(dataset), project=args.project, name="protocol_b", device=args.device,
        batch=train_batch, **config
    )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise FileNotFoundError(f"Trainer-selected checkpoint not found: {best}")
    selected = YOLO(str(best))
    for split in (selection_split, final_endpoint):
        selected.val(
        data=str(dataset),
        split=split,
        imgsz=imgsz,
        batch=eval_batch,
        conf=eval_conf,
        iou=eval_iou,
        max_det=eval_max_det,
        half=eval_half,
        augment=eval_augment,
        workers=workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
