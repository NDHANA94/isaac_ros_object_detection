#!/usr/bin/env python3

# ─────────────────────────────────────────────────────────────────────────────
# MIT License

# Copyright (c) 2026 WM Nipun Dhananjaya

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ───────────────────────────────────────────────────────────────────────────── 

# @file: inspect_model.py
# @brief: Utility script to inspect TensorRT engine (.plan / .engine) and ONNX (.onnx) files, 
#         printing details about input/output tensors, shapes, dtypes, and layer counts.
# @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
# @company: Sintez.LLC 
# @date: 2026-03-01


"""
inspect_model.py

Inspect a TensorRT engine (.plan / .engine) or ONNX (.onnx) file and print
details including input/output tensor names, shapes, dtypes, and layer count.

Usage:
  python3 inspect_model.py model.plan
  python3 inspect_model.py model.onnx
  python3 inspect_model.py model.plan --layers        # also list all TRT layers
  python3 inspect_model.py model.onnx --netron        # open in Netron browser
"""

import os
import sys
import argparse


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _hr(char='─', width=72):
    print(char * width)


def _section(title):
    _hr()
    print(f"  {title}")
    _hr()


# ──────────────────────────────────────────────────────────────────────────────
# TensorRT engine inspection
# ──────────────────────────────────────────────────────────────────────────────

_TRT_DTYPE_MAP = {
    0: 'float32',
    1: 'float16',
    2: 'int8',
    3: 'int32',
    4: 'bool',
    5: 'uint8',
    6: 'float8e4m3',
    7: 'bfloat16',
    8: 'int64',
    9: 'int4',
}


def _trt_mode_flag(config_flags: int) -> str:
    """Best-effort decode of builder flags from the engine context."""
    # We can't read flags from a deserialized engine — return placeholder.
    return "n/a (not stored in serialized engine)"


def inspect_trt(path: str, show_layers: bool):
    try:
        import tensorrt as trt
    except ImportError:
        sys.exit("[ERROR] tensorrt Python package not installed.")

    print(f"\n  TensorRT version : {trt.__version__}")
    print(f"  File             : {os.path.abspath(path)}")
    print(f"  Size             : {os.path.getsize(path) / 1e6:.2f} MB")

    logger  = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        sys.exit("[ERROR] Failed to deserialize engine. "
                 "Check TRT version compatibility.")

    # ── Basic info ──
    _section("Engine Info")
    trt_major = int(trt.__version__.split('.')[0])

    # num_io_tensors available TRT ≥ 8.5
    if hasattr(engine, 'num_io_tensors'):
        num_io = engine.num_io_tensors
        tensor_names = [engine.get_tensor_name(i) for i in range(num_io)]
        inputs  = [n for n in tensor_names
                   if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in tensor_names
                   if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    else:
        # TRT 8.x legacy API
        num_io  = engine.num_bindings
        inputs  = [engine.get_binding_name(i)
                   for i in range(num_io) if engine.binding_is_input(i)]
        outputs = [engine.get_binding_name(i)
                   for i in range(num_io) if not engine.binding_is_input(i)]

    num_layers = engine.num_layers if hasattr(engine, 'num_layers') else 'n/a'
    max_batch  = engine.max_batch_size if hasattr(engine, 'max_batch_size') else 'n/a'

    print(f"  Layers           : {num_layers}")
    print(f"  Max batch size   : {max_batch}")
    print(f"  I/O tensors      : {len(inputs)} input(s), {len(outputs)} output(s)")

    # ── Input tensors ──
    _section("Input Tensors")
    for name in inputs:
        if hasattr(engine, 'get_tensor_shape'):
            shape = tuple(engine.get_tensor_shape(name))
            dtype_enum = engine.get_tensor_dtype(name)
            dtype = _TRT_DTYPE_MAP.get(int(dtype_enum), str(dtype_enum))
            location = engine.get_tensor_location(name)
        else:
            idx   = engine[name]
            shape = tuple(engine.get_binding_shape(idx))
            dtype_enum = engine.get_binding_dtype(idx)
            dtype = _TRT_DTYPE_MAP.get(int(dtype_enum), str(dtype_enum))
            location = 'GPU'
        print(f"  name     : {name}")
        print(f"  shape    : {shape}")
        print(f"  dtype    : {dtype}")
        print(f"  location : {location}")
        print()

    # ── Output tensors ──
    _section("Output Tensors")
    for name in outputs:
        if hasattr(engine, 'get_tensor_shape'):
            shape = tuple(engine.get_tensor_shape(name))
            dtype_enum = engine.get_tensor_dtype(name)
            dtype = _TRT_DTYPE_MAP.get(int(dtype_enum), str(dtype_enum))
            location = engine.get_tensor_location(name)
        else:
            idx   = engine[name]
            shape = tuple(engine.get_binding_shape(idx))
            dtype_enum = engine.get_binding_dtype(idx)
            dtype = _TRT_DTYPE_MAP.get(int(dtype_enum), str(dtype_enum))
            location = 'GPU'
        print(f"  name     : {name}")
        print(f"  shape    : {shape}")
        print(f"  dtype    : {dtype}")
        print(f"  location : {location}")
        print()

    # ── Layer list ──
    if show_layers and hasattr(engine, 'num_layers') and engine.num_layers:
        _section(f"All Layers  ({engine.num_layers})")
        inspector = engine.create_engine_inspector()
        if inspector:
            for i in range(engine.num_layers):
                info = inspector.get_layer_information(
                    i, trt.LayerInformationFormat.ONELINE)
                print(f"  [{i:4d}] {info}")
        else:
            print("  (engine inspector not available on this TRT version)")

    _hr()


# ──────────────────────────────────────────────────────────────────────────────
# ONNX inspection
# ──────────────────────────────────────────────────────────────────────────────

_ONNX_DTYPE_MAP = {
    0:  'undefined',
    1:  'float32',
    2:  'uint8',
    3:  'int8',
    4:  'uint16',
    5:  'int16',
    6:  'int32',
    7:  'int64',
    8:  'string',
    9:  'bool',
    10: 'float16',
    11: 'float64',
    12: 'uint32',
    13: 'uint64',
    14: 'complex64',
    15: 'complex128',
    16: 'bfloat16',
    17: 'float8e4m3fn',
    18: 'float8e4m3fnuz',
    19: 'float8e5m2',
    20: 'float8e5m2fnuz',
}


def _dim_to_str(dim):
    """Convert an ONNX TensorShapeProto Dimension to a readable string."""
    which = dim.WhichOneof('value')
    if which == 'dim_value':
        return str(dim.dim_value)
    elif which == 'dim_param':
        return dim.dim_param or '?'
    return '?'


def _onnx_tensor_info(value_info) -> dict:
    t = value_info.type.tensor_type
    shape = [_dim_to_str(d) for d in t.shape.dim] if t.HasField('shape') else ['?']
    return {
        'name' : value_info.name,
        'shape': shape,
        'dtype': _ONNX_DTYPE_MAP.get(t.elem_type, f"unknown({t.elem_type})"),
    }


def inspect_onnx(path: str, show_layers: bool):
    try:
        import onnx
    except ImportError:
        sys.exit("[ERROR] onnx not installed.  pip install onnx")

    model = onnx.load(path)
    graph = model.graph

    print(f"\n  ONNX opset       : {', '.join(str(v.version) for v in model.opset_import)}")
    print(f"  IR version       : {model.ir_version}")
    print(f"  Producer         : {model.producer_name or 'n/a'} "
          f"{model.producer_version or ''}")
    print(f"  Domain           : {model.domain or 'n/a'}")
    print(f"  Model version    : {model.model_version}")
    print(f"  Doc string       : {model.doc_string[:80] if model.doc_string else 'n/a'}")
    print(f"  File             : {os.path.abspath(path)}")
    print(f"  Size             : {os.path.getsize(path) / 1e6:.2f} MB")
    print(f"  Nodes (ops)      : {len(graph.node)}")

    # ── Input tensors ──
    _section("Input Tensors")
    for vi in graph.input:
        info = _onnx_tensor_info(vi)
        print(f"  name  : {info['name']}")
        print(f"  shape : {info['shape']}")
        print(f"  dtype : {info['dtype']}")
        print()

    # ── Output tensors ──
    _section("Output Tensors")
    for vi in graph.output:
        info = _onnx_tensor_info(vi)
        print(f"  name  : {info['name']}")
        print(f"  shape : {info['shape']}")
        print(f"  dtype : {info['dtype']}")
        print()

    # ── Intermediate value info ──
    if graph.value_info:
        _section(f"Intermediate Tensors  ({len(graph.value_info)})")
        for vi in graph.value_info:
            info = _onnx_tensor_info(vi)
            print(f"  {info['name']:40s}  shape={info['shape']}  dtype={info['dtype']}")

    # ── Initializers (weights) summary ──
    init_sizes = {}
    for init in graph.initializer:
        total = 1
        for d in init.dims:
            total *= d
        init_sizes[init.name] = (list(init.dims), total * 4 / 1e6)  # rough MB
    total_params = sum(v[1] for v in init_sizes.values())
    _section(f"Weights (initializers)  —  {len(init_sizes)} tensors, "
             f"~{total_params:.1f} MB")
    for name, (shape, mb) in list(init_sizes.items())[:20]:
        print(f"  {name:50s}  {str(shape):30s}  ~{mb:.3f} MB")
    if len(init_sizes) > 20:
        print(f"  ... {len(init_sizes) - 20} more weights not shown (use --layers)")

    # ── Op type histogram ──
    _section("Op Type Histogram")
    from collections import Counter
    op_counts = Counter(n.op_type for n in graph.node)
    for op, count in op_counts.most_common():
        print(f"  {op:30s}  {count:>5}")

    # ── Layer list ──
    if show_layers:
        _section(f"All Nodes  ({len(graph.node)})")
        for i, node in enumerate(graph.node):
            inputs  = ', '.join(node.input)
            outputs = ', '.join(node.output)
            print(f"  [{i:4d}] {node.op_type:20s}  "
                  f"in=[{inputs}]  out=[{outputs}]")

    _hr()


# ──────────────────────────────────────────────────────────────────────────────
# Netron launcher
# ──────────────────────────────────────────────────────────────────────────────

def launch_netron(path: str):
    """Open the model in Netron for interactive visualisation."""
    try:
        import netron
        print(f"\n[INFO] Opening {path} in Netron...")
        netron.start(path)
    except ImportError:
        print("\n[INFO] Netron not installed.  pip install netron")
        print(f"[INFO] Alternatively open:  https://netron.app  and drag {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Inspect a TensorRT engine or ONNX model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('model',
                        help='Path to .plan / .engine (TensorRT) or .onnx file')
    parser.add_argument('--layers', action='store_true',
                        help='Also list all layers / nodes')
    parser.add_argument('--netron', action='store_true',
                        help='Open the model in Netron after inspection')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"[ERROR] File not found: {args.model}")

    ext = os.path.splitext(args.model)[1].lower()

    _hr('═')
    print(f"  Model Inspector  —  {os.path.basename(args.model)}")
    _hr('═')

    if ext in ('.plan', '.engine', '.trt'):
        inspect_trt(args.model, show_layers=args.layers)
    elif ext == '.onnx':
        inspect_onnx(args.model, show_layers=args.layers)
    else:
        sys.exit(f"[ERROR] Unsupported extension '{ext}'. "
                 "Expected .plan, .engine, .trt, or .onnx")

    if args.netron:
        launch_netron(args.model)


if __name__ == '__main__':
    main()
