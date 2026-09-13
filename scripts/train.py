#!/usr/bin/env python3
"""Train SMFR-YOLO from the publication configuration."""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[1]


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs/train.yaml"))
    parser.add_argument("--data", help="Optional dataset YAML override")
    parser.add_argument("--device", help="Optional device override")
    parser.add_argument("--project", help="Optional output directory override")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_path = resolve_repo_path(config.pop("model"))
    configured_data = config.pop("data")
    data_path = Path(args.data) if args.data else resolve_repo_path(configured_data)
    if args.device is not None:
        config["device"] = args.device
    if args.project is not None:
        config["project"] = args.project
    YOLO(str(model_path)).train(data=str(data_path), name="train", **config)


if __name__ == "__main__":
    main()
