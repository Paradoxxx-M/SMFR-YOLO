#!/usr/bin/env python3
"""Export through the pinned DeepStream-Yolo path and verify its ONNX contract."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from check_onnx_contract import validate_contract


REPO = Path(__file__).resolve().parents[1]
DEEPSTREAM_YOLO_COMMIT = "2894babce8e75c49115dbe0c7b516289ed853565"


def checked_head(repository):
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--deepstream-yolo-root", required=True, type=Path)
    parser.add_argument("--config", default=REPO / "deployment/nano.yaml", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    deepstream_yolo = args.deepstream_yolo_root.resolve()
    exporter = deepstream_yolo / "utils/export_yoloV8.py"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not exporter.is_file():
        raise FileNotFoundError(exporter)
    head = checked_head(deepstream_yolo)
    if head != DEEPSTREAM_YOLO_COMMIT:
        raise RuntimeError(f"DeepStream-Yolo must be at {DEEPSTREAM_YOLO_COMMIT}; found {head}")
    exporter_status = subprocess.check_output(
        ["git", "-C", str(deepstream_yolo), "status", "--porcelain", "--", "utils/export_yoloV8.py"],
        text=True,
    ).strip()
    if exporter_status:
        raise RuntimeError(f"Pinned exporter has local modifications: {exporter_status}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["imgsz"] != 640 or config["batch"] != 1 or config["onnx_opset"] != 12:
        raise ValueError("Nano export contract must remain imgsz=640, batch=1, opset=12")
    command = [
        sys.executable,
        str(exporter),
        "-w",
        str(checkpoint),
        "-s",
        "640",
        "640",
        "--opset",
        "12",
        "--batch",
        "1",
    ]
    print("exporter=DeepStream-Yolo/utils/export_yoloV8.py")
    print(f"deepstream_yolo_commit={head}")
    subprocess.run(command, cwd=checkpoint.parent, check=True)
    onnx_path = checkpoint.with_suffix(".onnx")
    validate_contract(onnx_path)
    print(f"onnx={onnx_path}")


if __name__ == "__main__":
    main()
