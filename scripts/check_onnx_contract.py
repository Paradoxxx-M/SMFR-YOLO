#!/usr/bin/env python3
"""Validate the fixed ONNX contract used by the Jetson Nano deployment."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto


EXPECTED_INPUT_SHAPE = [1, 3, 640, 640]
EXPECTED_OUTPUT_SHAPE = [1, 8400, 6]
EXPECTED_OPSET = 12


def tensor_shape(value_info):
    dims = value_info.type.tensor_type.shape.dim
    return [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in dims]


def validate_contract(path):
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError(f"Expected one input and one output; found {len(model.graph.input)} and {len(model.graph.output)}")

    graph_input = model.graph.input[0]
    graph_output = model.graph.output[0]
    input_shape = tensor_shape(graph_input)
    output_shape = tensor_shape(graph_output)
    opsets = {entry.domain or "ai.onnx": entry.version for entry in model.opset_import}
    default_opset = opsets.get("ai.onnx")
    input_dtype = graph_input.type.tensor_type.elem_type
    output_dtype = graph_output.type.tensor_type.elem_type

    if input_shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"Input shape mismatch: expected {EXPECTED_INPUT_SHAPE}, found {input_shape}")
    if output_shape != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(f"Output shape mismatch: expected {EXPECTED_OUTPUT_SHAPE}, found {output_shape}")
    if input_dtype != TensorProto.FLOAT or output_dtype != TensorProto.FLOAT:
        raise ValueError(f"Expected FLOAT input/output; found {input_dtype}/{output_dtype}")
    if default_opset != EXPECTED_OPSET:
        raise ValueError(f"ONNX opset mismatch: expected {EXPECTED_OPSET}, found {default_opset}")

    print("onnx_checker=PASS")
    print(f"input_shape={input_shape} input_dtype=float")
    print(f"output_shape={output_shape} output_dtype=float")
    print(f"opset={default_opset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", type=Path)
    args = parser.parse_args()
    validate_contract(args.onnx_path)


if __name__ == "__main__":
    main()
