#!/usr/bin/env python3
"""Evaluate a checkpoint with the formal primary protocol."""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(REPO / "configs/evaluate.yaml"))
    parser.add_argument("--data", help="Optional dataset YAML override")
    parser.add_argument("--split", choices=("val", "test"), help="Optional endpoint override")
    parser.add_argument("--device", help="Optional device override")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    configured_data = config.pop("data")
    data = args.data or configured_data
    data_path = Path(data)
    if not data_path.is_absolute():
        data_path = REPO / data_path
    if args.split is not None:
        config["split"] = args.split
    if args.device is not None:
        config["device"] = args.device
    YOLO(args.checkpoint).val(data=str(data_path), **config)


if __name__ == "__main__":
    main()
