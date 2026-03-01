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

# @file: export_model_plan.py
# @brief: Given a YOLO .pt model, export to ONNX and then build a TensorRT engine (.plan)
# @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
# @company: Sintez.LLC 
# @date: 2026-03-01


"""
export_model_plan.py

Export a YOLO .pt model → ONNX → TensorRT engine (.plan)

Supported quantization modes:
  fp32  — full precision
  fp16  — half precision  (recommended for Jetson Orin, ~2× faster)
  int8  — INT8 quantized  (fastest / smallest, ~4× vs fp32, needs calib images)
  int88 — alias for int8  (INT8 weights + INT8 activations)

Examples:
  python3 export_model_plan.py --model yolo26n.pt
  python3 export_model_plan.py --model yolo26n.pt --quantization fp16
  python3 export_model_plan.py --model yolo26n.pt --quantization int8 --calib_dir /data/coco_calib
  python3 export_model_plan.py --model yolo26n.pt --quantization int88 --calib_dir /data/coco_calib
"""

import os
import sys
import glob
import argparse
import numpy as np

FILE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(FILE_DIR, '..', 'engines')
os.makedirs(MODELS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# INT8 Entropy Calibrator (IInt8EntropyCalibrator2)
# ──────────────────────────────────────────────────────────────────────────────

def _make_calibrator(image_dir, imgsz, batch_size, cache_file):
    """
    Dynamically subclass trt.IInt8EntropyCalibrator2 so we can import TRT
    at runtime without issues at module-load time.
    """
    import tensorrt as trt
    import pycuda.driver as cuda
    try:
        import pycuda.autoinit  # noqa: F401
    except Exception:
        pass

    class _Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            import cv2  # noqa: F401
            self._cv2       = cv2
            self._cuda      = cuda
            self.imgsz      = imgsz
            self.batch_size = batch_size
            self.cache_file = cache_file
            self.current_index = 0

            exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
            self.images = []
            for ext in exts:
                self.images.extend(glob.glob(os.path.join(image_dir, ext)))
            if not self.images:
                raise FileNotFoundError(
                    f"[Calibrator] No images in: {image_dir}")
            print(f"[Calibrator] {len(self.images)} images, "
                  f"batch={batch_size}")

            nbytes = (batch_size * 3 * imgsz * imgsz
                      * np.dtype(np.float32).itemsize)
            self.device_input = cuda.mem_alloc(nbytes)

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            cv2 = self._cv2
            if self.current_index + self.batch_size > len(self.images):
                return None
            batch = []
            for i in range(self.batch_size):
                img = cv2.imread(self.images[self.current_index + i])
                img = cv2.resize(img, (self.imgsz, self.imgsz))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(
                    np.float32) / 255.0
                batch.append(np.transpose(img, (2, 0, 1)))  # HWC → CHW
            arr = np.ascontiguousarray(np.stack(batch, axis=0))
            self._cuda.memcpy_htod(self.device_input, arr)
            self.current_index += self.batch_size
            done = self.current_index // self.batch_size
            total = len(self.images) // self.batch_size
            print(f"[Calibrator] batch {done}/{total}")
            return [self.device_input]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                print(f"[Calibrator] Loading cache: {self.cache_file}")
                with open(self.cache_file, 'rb') as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_file, 'wb') as f:
                f.write(cache)
            print(f"[Calibrator] Cache saved: {self.cache_file}")

    return _Calibrator()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Export YOLO .pt → ONNX → TensorRT engine (.plan)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLO weights, e.g. yolo26n.pt')
    parser.add_argument('--quantization', type=str,
                        choices=['fp32', 'fp16', 'int8', 'int88'],
                        default='fp16',
                        help='Quantization mode. '
                             'int88 = alias for int8 '
                             '(INT8 weights + INT8 activations).')
    parser.add_argument('--engine', type=str, default=None,
                        help='Output .plan path '
                             '(default: models/<stem>_<quant>.plan)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size (square)')
    parser.add_argument('--workspace_gb', type=float, default=4.0,
                        help='TensorRT workspace memory in GB')
    parser.add_argument('--calib_dir', type=str, default=None,
                        help='Image directory for INT8/INT88 calibration')
    parser.add_argument('--calib_cache', type=str, default=None,
                        help='Path to save/load INT8 calibration cache '
                             '(default: models/<stem>_int8.cache)')
    parser.add_argument('--calib_batch', type=int, default=8,
                        help='Calibration batch size')
    parser.add_argument('--keep_onnx', action='store_true',
                        help='Keep the intermediate ONNX file')
    args = parser.parse_args()

    # ── Prime CUDA context early ─────────────────────────────────────────────
    # On Jetson (unified/shared memory) TensorRT initialises CUDA at *import*
    # time. Ultralytics may import TRT internally during model.export().
    # Prime CUDA via torch first so TRT doesn't abort with
    # "terminate called without an active exception".
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.init()
            _dummy = _torch.zeros(1, device='cuda')
            del _dummy
            print("[INFO] CUDA context primed via torch.")
        else:
            print("[WARN] torch.cuda not available — TRT may fail to init.")
    except Exception as _exc:
        print(f"[WARN] Could not prime CUDA: {_exc}")

    # ── Normalise quantization: int88 → int8 ────────────────────────────────
    quant = args.quantization.lower()
    if quant == 'int88':
        quant = 'int8'
        print("[INFO] --quantization int88 → INT8 "
              "(INT8 weights + INT8 activations).")

    # ── Derive default output path ───────────────────────────────────────────
    model_stem = os.path.splitext(os.path.basename(args.model))[0]
    if args.engine is None:
        args.engine = os.path.join(MODELS_DIR, f"{model_stem}_{quant}.plan")

    print(f"[INFO] Model       : {args.model}")
    print(f"[INFO] Quantization: {quant}")
    print(f"[INFO] Image size  : {args.imgsz}x{args.imgsz}")
    print(f"[INFO] Workspace   : {args.workspace_gb} GB")
    print(f"[INFO] Output      : {args.engine}")

    # ── 1. Export ONNX via ultralytics ───────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics not installed.  pip install ultralytics")

    print("\n[STEP 1] Exporting ONNX via ultralytics...")
    yolo_model    = YOLO(args.model)
    export_result = yolo_model.export(
        format='onnx',
        imgsz=args.imgsz,
        dynamic=False,
        simplify=True,
        opset=17,
    )

    # ultralytics ≥ 8.x returns the output path as a string
    if isinstance(export_result, str) and os.path.isfile(export_result):
        onnx_path = export_result
    else:
        # Fallback: ultralytics saves the .onnx next to the .pt
        onnx_path = os.path.splitext(os.path.abspath(args.model))[0] + '.onnx'

    if not os.path.isfile(onnx_path):
        sys.exit(f"[ERROR] ONNX export failed — not found: {onnx_path}")
    print(f"[INFO] ONNX saved  : {onnx_path}")

    # ── 2. Parse ONNX with TensorRT ──────────────────────────────────────────
    print("\n[STEP 2] Parsing ONNX with TensorRT...")
    try:
        import tensorrt as trt
    except ImportError:
        sys.exit("[ERROR] tensorrt Python bindings not found.")

    print(f"\n[STEP 2] Parsing ONNX with TensorRT {trt.__version__}...")
    TRT_LOGGER  = trt.Logger(trt.Logger.INFO)
    builder     = trt.Builder(TRT_LOGGER)

    # TRT 10.x: EXPLICIT_BATCH is the default and the flag is deprecated.
    # TRT 8.x: flag must be set explicitly.
    trt_major = int(trt.__version__.split('.')[0])
    if trt_major >= 10:
        network = builder.create_network()
    else:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not onnx_parser.parse(f.read()):
            print("[ERROR] ONNX parse failed:")
            for i in range(onnx_parser.num_errors):
                print(f"  {onnx_parser.get_error(i)}")
            sys.exit(1)

    print(f"[INFO] Network inputs : {network.num_inputs}")
    print(f"[INFO] Network outputs: {network.num_outputs}")

    # ── 3. Builder config (TRT 10.x API) ────────────────────────────────────
    print("\n[STEP 3] Configuring TensorRT builder...")
    config = builder.create_builder_config()

    # Workspace — replaces deprecated builder.max_workspace_size
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(args.workspace_gb * (1 << 30)))

    if quant == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 enabled.")
        else:
            print("[WARN] FP16 not supported on this platform — using FP32.")

    elif quant == 'int8':
        if not builder.platform_has_fast_int8:
            print("[WARN] Platform may not have fast INT8 support.")
        config.set_flag(trt.BuilderFlag.INT8)
        # Keep FP16 active as fallback for layers that don't support INT8
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] INT8 enabled (+ FP16 fallback for unsupported layers).")

        if args.calib_dir:
            calib_cache = args.calib_cache or os.path.join(
                MODELS_DIR, f"{model_stem}_int8.cache")
            try:
                config.int8_calibrator = _make_calibrator(
                    image_dir  = args.calib_dir,
                    imgsz      = args.imgsz,
                    batch_size = args.calib_batch,
                    cache_file = calib_cache,
                )
            except ImportError:
                sys.exit("[ERROR] pycuda required for INT8 calibration.\n"
                         "        pip install pycuda")
            print(f"[INFO] Calibration dir  : {args.calib_dir}")
            print(f"[INFO] Calibration cache: {calib_cache}")
        else:
            print("[WARN] No --calib_dir provided for INT8.")
            print("[WARN] TensorRT will use implicit quantization — "
                  "accuracy may be lower.")
            print("[WARN] For best results: "
                  "--calib_dir <folder_of_representative_images>")

    # ── 4. Build serialized engine (TRT 10.x replaces build_cuda_engine) ────
    print("\n[STEP 4] Building TensorRT engine "
          "(first run can take 5–15 min on Orin)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        sys.exit("[ERROR] TensorRT engine build failed. "
                 "Check log output above.")

    os.makedirs(os.path.dirname(os.path.abspath(args.engine)), exist_ok=True)
    with open(args.engine, 'wb') as f:
        f.write(serialized)

    size_mb = os.path.getsize(args.engine) / 1e6
    print(f"\n[OK] Engine saved : {args.engine}  ({size_mb:.1f} MB)")

    if not args.keep_onnx and os.path.isfile(onnx_path):
        os.remove(onnx_path)
        print(f"[INFO] Removed temp ONNX: {onnx_path}")


if __name__ == '__main__':
    main()
