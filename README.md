# isaac_ros_object_detection

A ROS 2 composable package that delivers **GPU-accelerated YOLO object detection** on NVIDIA Jetson
platforms. The pipeline ingests camera frames via NITROS zero-copy transport, pre-processes them
with the Jetson VIC hardware engine or CUDA, runs inference through a TensorRT FP16 engine, and
publishes standard `vision_msgs/Detection2DArray` detections — plus an optional GPU-annotated
preview image.

> **Part of** the [`ros2_px4_object_follower`](../../..) meta-package.  
> Designed to feed directly into the `object_tracker` node downstream.

---

## Table of Contents

- [Overview](#overview)
- [Full Pipeline Data Flow](#full-pipeline-data-flow)
- [Nodes](#nodes)
- [Hardware Acceleration & CPU Fallback](#hardware-acceleration--cpu-fallback)
- [Package Structure](#package-structure)
- [Dependencies](#dependencies)
- [Building](#building)
- [TensorRT Engine Preparation](#tensorrt-engine-preparation)
- [Topics](#topics)
- [Parameters](#parameters)
- [Launch](#launch)
- [Tuning Guide](#tuning-guide)
- [License](#license)

---

## Overview

| Property | Value |
|---|---|
| Package name | `isaac_ros_object_detection` |
| Version | `0.0.0` |
| License | MIT |
| Author | WM Nipun Dhananjaya — nipun.dhananjaya@gmail.com |
| Target hardware | NVIDIA Jetson Orin (sm_87 / Ampere) |

**Two composable nodes** are provided in a single shared library:

| Node class | Plugin name | Role |
|---|---|---|
| `ObjectDetectionNode` | `nvidia::isaac_ros::object_detection::ObjectDetectionNode` | TensorRT inference + detection publisher |
| `AnnotatedImagePublisherNode` | `nvidia::isaac_ros::object_detection::AnnotatedImagePublisherNode` | GPU annotation + compressed image publisher |

---

## Full Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Camera (ArducamB0573Node / any V4L2 driver)                            │
│  Publishes: NitrosImage  (NV12 / RGB8) — GPU-resident, zero-copy        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │  /arducam/left/nitros_image_rgb8
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DnnImageEncoderNode  (isaac_ros_dnn_image_encoder — upstream pkg)      │
│                                                                         │
│  • Resizes frame to network input size (640×640)                        │
│  • Normalises pixel values: (pixel / 255 - mean) / stddev               │
│  • Converts HWC → NCHW float32 tensor                                   │
│  • Keeps tensor GPU-resident via NITROS GXF memory pool                 │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │  /dnn_enc/encoded_tensor  (NitrosTensorList)
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ObjectDetectionNode  ← THIS PACKAGE                                    │
│                                                                         │
│  1. ManagedNitrosSubscriber receives GPU tensor pointer (no host copy)  │
│                                                                         │
│  2. Pre-process  ─── hardware_decoder param ─────────────────────────┐  │
│     "cuda"  → cudaMemcpyDeviceToDevice into TRT input buffer         │  │
│     "vic"   → NvBufSurfTransform: YUV/NV12→RGBA + bilinear resize    │  │
│               then CUDA kernel: RGBA→float32 NCHW normalisation      │  │
│     "auto"  → same as "vic"                                          │  │
│                                                                      │  │
│  3. TensorRT inference  (FP16 engine, fully async on CUDA stream)    │  │
│     enqueueV3() returns immediately; CPU never stalls              ◄─┘  │
│                                                                         │
│  4. Decode output  (GPU→CPU DMA transfer, then CPU parse)               │
│     yolo26n:  NMS embedded in model → output (1,300,6) float32          │
│               [x1,y1,x2,y2, confidence, class_id]  xyxy @ 640×640       │
│     Filters:  confidence_threshold, class whitelist                     │
│     Converts: xyxy → centre+size for Detection2DArray                   │
│                                                                         │
│  5. Publish  →  vision_msgs/Detection2DArray                            │
└──────────┬───────────────────────────────────────────────────────────┬──┘
           │  /object_detection/detections                             │
           │                                                           │ (same topic)
           ▼                                                           ▼
┌─────────────────────────────┐       ┌─────────────────────────────────────┐
│  object_tracker node        │       │  AnnotatedImagePublisherNode        │
│  (separate package)         │       │  ← THIS PACKAGE                     │
│  BYTETracker → track IDs    │       │                                     │
└─────────────────────────────┘       │  Subscribes to:                     │
                                      │    • NitrosImage / raw / compressed │
                                      │    • Detection2DArray               │
                                      │    • CameraInfo                     │
                                      │                                     │
                                      │  GPU annotation pipeline:           │
                                      │  VIC: NitrosImage→RGBA colour conv  │
                                      │  CUDA: bbox rect + glyph-atlas text │
                                      │  nvJPEG: JPEG encode on GPU         │
                                      │  CPU fallback: OpenCV JPEG encode   │
                                      │                                     │
                                      │  Publishes:                         │
                                      │  /object_detection/annotated_image  │
                                      │  (compressed or raw)                │
                                      └─────────────────────────────────────┘
```

---

## Nodes

### ObjectDetectionNode

The core inference node. On startup it:

1. Deserializes a pre-built TensorRT `.plan` engine from disk.
2. Allocates persistent GPU input / output buffers (`d_input_`, `d_output_`) — reused every frame,
   no per-frame allocation.
3. Optionally allocates a `NvBufSurface` workspace for the VIC pre-processing path.
4. Creates a `ManagedNitrosSubscriber` that keeps the incoming tensor GPU-resident via the NITROS
   GXF memory pool — **the tensor data never touches host RAM**.

Per-frame callback (`InputCallback`):
- Serialised with a `std::mutex` because `component_container_mt` uses a multi-threaded executor.
- Branches into VIC or CUDA pre-processing based on `hardware_decoder`.
- Calls `trt_context_->enqueueV3()` — fully asynchronous; the CUDA stream serialises VIC output →
  TRT kernel automatically.
- Synchronises the stream once inference is done, DMAs the output buffer to host, and parses the
  detection rows.
- Converts xyxy absolute-pixel boxes back to centre+size format and publishes.

### AnnotatedImagePublisherNode

A visualisation-only node that draws detection boxes and class labels on the raw camera frame.
It subscribes independently to the raw image, detections, and camera info — it does **not** sit
in the inference callback — so annotation latency never blocks detection throughput.

Hardware acceleration inside this node:

| Step | Engine |
|---|---|
| Colour-format conversion (NitrosImage → RGBA) | VIC (`NvBufSurfTransform`) |
| Bounding-box rectangle drawing | CUDA kernel (`annotation_kernels.cu`) |
| Text / label stamping | CUDA glyph-atlas kernel (atlas built once with OpenCV at startup) |
| JPEG encode for compressed publish | nvJPEG (GPU) with OpenCV CPU fallback |

---

## Hardware Acceleration & CPU Fallback

### Pre-processing path selection (`hardware_decoder` parameter)

```
hardware_decoder = "cuda"   ← recommended when upstream is DnnImageEncoder
      │
      │   Input: float32 NCHW tensor already in GPU memory (NITROS)
      │   Action: cudaMemcpyDeviceToDevice into TRT input buffer
      │   Cost: ~0 µs — single DMA copy within the GPU
      │
      └─► TRT input buffer ready

hardware_decoder = "vic" / "auto"   ← use when upstream is ArgusCamera direct
      │
      │   Input: NvBufSurface* pointer to a raw YUV/NV12 GPU buffer
      │   Step 1 — VIC engine (NvBufSurfTransform):
      │             YUV/NV12 → RGBA  +  bilinear resize to 640×640
      │             Runs entirely on the dedicated VIC hardware block,
      │             zero CPU involvement, output stays in GPU memory.
      │   Step 2 — CUDA kernel (cuda_normalize_rgba):
      │             RGBA [0–255]  →  float32 [0.0–1.0]  NCHW layout
      │             Trivial element-wise op, < 0.1 ms on Orin GPU
      │
      └─► TRT input buffer ready
```

### Why the `"cuda"` path is preferred with DnnImageEncoder

`DnnImageEncoderNode` (upstream) already performs resize + normalisation + NCHW conversion and
publishes the result as a `NitrosTensorList` stored in a GXF GPU memory pool. When
`ObjectDetectionNode` receives this tensor via `ManagedNitrosSubscriber`, the buffer is already
in the exact format TensorRT expects. A single `cudaMemcpyDeviceToDevice` places it into the
pre-allocated `d_input_` buffer — **no VIC allocation, no RGBA intermediate, no CPU work**.

### CUDA stream non-blocking design

A single `cudaStreamNonBlocking` stream is created at node startup and used for all per-frame GPU
work: VIC normalisation kernel → TRT `enqueueV3()`. Because work items are queued on the same
stream they execute in order, so there is no need for explicit CUDA events between steps. The CPU
only synchronises (`cudaStreamSynchronize`) **once per frame**, just before reading the output
buffer back to host for parsing.

### nvJPEG vs OpenCV fallback (AnnotatedImagePublisherNode)

At build time the `CMakeLists.txt` searches for the `nvjpeg` library:

```cmake
find_library(NVJPEG_LIB nvjpeg ...)
if(NOT NVJPEG_LIB)
  message(WARNING "nvjpeg not found; GPU annotation path will be disabled")
endif()
```

- **nvJPEG present** → `WITH_NVJPEG` compile definition is set; JPEG encoding of the annotated
  frame runs entirely on the GPU. RGBA buffer flows directly from the annotation CUDA kernel into
  `nvjpegEncodeImage()` without any host copy.
- **nvJPEG absent** → the annotated RGBA buffer is copied to host and encoded with
  `cv::imencode()` (OpenCV CPU). Functional but ~5–15 ms slower per frame at 1080p.

---

## Package Structure

```
isaac_ros_object_detection/
├── CMakeLists.txt
├── package.xml
├── config/
│   ├── obj_det_params.yaml              # ObjectDetectionNode parameters
│   ├── annotated_img_publisher_params.yaml  # AnnotatedImagePublisherNode parameters
│   ├── dnn_img_enc_params.yaml          # DnnImageEncoder parameters (upstream node)
│   └── coco.names                       # 80-class COCO label file
├── engines/
│   ├── yolo26n_fp16.plan                # TensorRT FP16 engine — NMS embedded, output (1,300,6)
│   └── yolov8n_fp16.plan                # Alternative YOLOv8n FP16 engine
├── include/
│   └── isaac_ros_object_detection/
│       ├── object_detection_node.hpp    # ObjectDetectionNode + Object struct + TRTLogger
│       ├── annotated_image_publisher_node.hpp
│       └── annotation_kernels.hpp       # CUDA annotation kernel declarations
├── launch/
│   ├── composible_node_container.launch.py  # Full 4-node pipeline (recommended)
│   ├── annotated_image_publisher.launch.py  # Annotation node only
│   └── dev.launch.py                    # Development / debug launch
├── scripts/
│   ├── export_model_plan.py             # Export YOLO → ONNX → TensorRT .plan
│   └── inspect_model.py                 # Inspect TRT engine bindings
└── src/
    ├── main.cpp                         # Standalone executable entry point
    ├── object_detection_node.cpp        # Core inference node implementation
    ├── annotated_image_publisher_node.cpp
    ├── cuda_kernels.cu                  # NV12 pre-process + RGBA normalise kernels
    └── annotation_kernels.cu            # Bbox drawing + glyph-atlas text kernels
```

---

## Dependencies

### ROS 2 / Isaac ROS

| Dependency | Purpose |
|---|---|
| `rclcpp`, `rclcpp_components` | ROS 2 C++ client library + composable nodes |
| `sensor_msgs`, `vision_msgs`, `visualization_msgs` | Image, detection, marker message types |
| `isaac_ros_managed_nitros` | `ManagedNitrosSubscriber` — GPU-resident zero-copy transport |
| `isaac_ros_nitros_tensor_list_type` | `NitrosTensorListView` — typed GPU tensor view |
| `isaac_ros_nitros_image_type` | `NitrosImageView` — typed GPU image view |

### System / Hardware

| Dependency | Purpose |
|---|---|
| CUDA Toolkit | `cuda_runtime_api.h`, async streams, device memory |
| TensorRT (`nvinfer`, `nvinfer_plugin`) | Engine deserialisation and inference |
| `nvbufsurface` / `nvbufsurftransform` | VIC hardware engine for pre-processing |
| nvJPEG (`nvjpeg`) | GPU JPEG encode in `AnnotatedImagePublisherNode` *(optional)* |
| OpenCV (`core`, `imgproc`, `imgcodecs`) | Glyph atlas generation at startup; CPU JPEG fallback |

Install system packages:

```bash
sudo apt install \
  ros-$ROS_DISTRO-vision-msgs \
  ros-$ROS_DISTRO-isaac-ros-managed-nitros \
  ros-$ROS_DISTRO-isaac-ros-nitros-tensor-list-type \
  ros-$ROS_DISTRO-isaac-ros-nitros-image-type \
  ros-$ROS_DISTRO-isaac-ros-dnn-image-encoder
```

---

## Building

```bash
cd ~/ros2_ws
colcon build --packages-select isaac_ros_object_detection \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

> **CUDA architecture:** The `CMakeLists.txt` sets `CUDA_ARCHITECTURES "87"` (Jetson Orin Nano /
> Orin NX — Ampere sm_87). Change this to match your Jetson model if needed:
> - Orin AGX / NX: `87`
> - Xavier: `72`
> - TX2: `62`

---

## TensorRT Engine Preparation

The node loads a pre-built `.plan` file — it does **not** compile the ONNX model at runtime.
You must export the engine once on the **target Jetson** (engines are not portable across GPU
architectures or TensorRT versions).

### Export a YOLO model to a TensorRT .plan

```bash
cd scripts/

# 1. Export PyTorch → ONNX → TensorRT FP16 (runs on Jetson)
python3 export_model_plan.py \
  --weights yolov8n.pt \
  --output ../engines/yolov8n_fp16.plan \
  --precision fp16 \
  --img-size 640
```

The included `engines/yolo26n_fp16.plan` was built from a YOLO26n model with embedded NMS.  
Its output binding is `(1, 300, 6)` float32 — each row is `[x1, y1, x2, y2, confidence, class_id]`
in absolute pixels at the 640×640 input resolution.

### Inspect engine bindings (verify tensor names before editing params.yaml)

```bash
python3 scripts/inspect_model.py --engine engines/yolo26n_fp16.plan
```

This prints all input/output binding names, shapes, and dtypes — use these to fill in
`engine.trt_input_binding_name` and `engine.output_tensor_name` in `obj_det_params.yaml`.

---

## Topics

### ObjectDetectionNode

#### Subscribed

| Topic | Type | Default | Notes |
|---|---|---|---|
| `/dnn_enc/encoded_tensor` | `NitrosTensorList` (NITROS) | `topics.enc_tensor_sub.topic_name` | Float32 NCHW GPU tensor from DnnImageEncoder. ManagedNitrosSubscriber appends `/nitros` suffix internally. |

#### Published

| Topic | Type | Default | Notes |
|---|---|---|---|
| `/object_detection/detections` | `vision_msgs/msg/Detection2DArray` | `topics.detection_pub.topic_name` | One `Detection2D` per detected object. Bounding box is centre + size in pixels at the **model input resolution** (640×640). `results[0].hypothesis.class_id` is the COCO class name string; `.score` is the detection confidence. |

### AnnotatedImagePublisherNode

#### Subscribed

| Topic | Type | Default |
|---|---|---|
| `/arducam/left/nitros_image_rgb8` | `NitrosImage` / `sensor_msgs/Image` / `CompressedImage` | `sub_topic_image.topic_name` |
| `/object_detection/detections` | `vision_msgs/msg/Detection2DArray` | `sub_topic_detections.topic_name` |
| `/arducam/left/image/camera_info` | `sensor_msgs/CameraInfo` | `sub_topic_camera_info.topic_name` |

#### Published

| Topic | Type | Default |
|---|---|---|
| `/object_detection/annotated_image` or `/object_detection/annotated_image/compressed` | `sensor_msgs/Image` or `sensor_msgs/CompressedImage` | `pub_topic_image.topic_name_prefix` + transport suffix |

---

## Parameters

### ObjectDetectionNode (`config/obj_det_params.yaml`)

#### Hardware

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hardware_decoder` | string | `"cuda"` | `"cuda"` — input is a float32 NCHW NITROS tensor (use with DnnImageEncoder). `"vic"` — input is a raw NvBufSurface YUV/NV12 (use with ArgusCamera direct path). `"auto"` — same as `"vic"`. |

#### Engine group (`engine.*`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine.model_path` | string | *(required)* | Absolute path to the TensorRT `.plan` file. Node shuts down with FATAL if empty. |
| `engine.class_names_path` | string | `""` | Path to a text file with one class name per line (e.g., `coco.names`). If empty, numeric class IDs are used as-is. |
| `engine.quantization` | string | `"fp16"` | Must match how the engine was exported: `"fp16"` or `"int8"`. Informational only — does not change runtime behaviour. |
| `engine.input_width` | int | `640` | Model input width in pixels. Must match the engine's input tensor shape. |
| `engine.input_height` | int | `640` | Model input height in pixels. |
| `engine.num_classes` | int | `80` | Number of output classes. Used for colour assignment in the annotator. |
| `engine.input_tensor_name` | string | `"input_tensor"` | Name of the NITROS tensor to extract from the incoming `NitrosTensorList`. Must match `final_tensor_name` in `dnn_img_enc_params.yaml`. |
| `engine.trt_input_binding_name` | string | `"images"` | TensorRT engine input binding name. Verify with `inspect_model.py`. |
| `engine.output_tensor_name` | string | `"output0"` | TensorRT engine output binding name. Verify with `inspect_model.py`. |

#### Filters group (`filters.*`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filters.confidence_threshold` | double | `0.20` | Minimum detection confidence to keep. Rows with a lower score are discarded during output decoding. |
| `filters.nms_threshold` | double | `0.45` | NMS IoU threshold. Informational for `yolo26n` (NMS is embedded in the model). Used if running a model that needs CPU-side NMS. |
| `filters.classes` | int[] | `[-1]` | Whitelist of class IDs to keep. Set to `[-1]` to keep all classes. Example: `[0]` keeps only class 0 (person). |

#### Topics / QoS (`topics.*`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `topics.enc_tensor_sub.topic_name` | string | `/dnn_enc/encoded_tensor` | NITROS tensor subscription topic |
| `topics.enc_tensor_sub.qos.reliability` | string | `best_effort` | `reliable` or `best_effort` |
| `topics.enc_tensor_sub.qos.durability` | string | `volatile` | `volatile` or `transient_local` |
| `topics.enc_tensor_sub.qos.depth` | int | `10` | History depth |
| `topics.detection_pub.topic_name` | string | `/object_detection/detections` | Detection publisher topic |
| `topics.detection_pub.qos.*` | — | same as above | Same QoS fields as subscriber |

### AnnotatedImagePublisherNode (`config/annotated_img_publisher_params.yaml`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sub_topic_image.topic_name` | string | `/arducam/left/nitros_image_rgb8` | Raw image subscription topic |
| `sub_topic_image.transport` | string | `"nitros"` | Input image transport: `"raw"`, `"compressed"`, or `"nitros"` |
| `sub_topic_detections.topic_name` | string | `/object_detection/detections` | Detection subscription topic |
| `sub_topic_camera_info.topic_name` | string | `/arducam/left/image/camera_info` | CameraInfo topic for bbox scaling |
| `pub_topic_image.topic_name_prefix` | string | `/object_detection/annotated_image` | Output topic prefix; transport suffix is appended |
| `pub_topic_image.transport` | string | `"compressed"` | Output transport: `"raw"` or `"compressed"` |
| `detection_model_input_width` | int | `640` | Must match the detection model input width — used to scale boxes from model space to image space |
| `detection_model_input_height` | int | `640` | Must match the detection model input height |
| `num_classes` | int | `80` | Used for per-class colour assignment when `box_color` is `[-1,-1,-1]` |
| `draw_labels` | bool | `true` | Overlay class name label above each box |
| `draw_confidence` | bool | `true` | Append confidence score to label text |
| `confidence_threshold` | double | `0.5` | Minimum confidence to draw a box (independent of the detector threshold) |
| `font_size` | double | `0.5` | Text scale factor |
| `font_thickness` | int | `1` | Text stroke thickness |
| `box_thickness` | int | `3` | Bounding-box rectangle line thickness |
| `box_color` | int[3] | `[-1,-1,-1]` | BGR colour for all boxes. Set to `[-1,-1,-1]` to use automatically assigned per-class colours (HSV hue cycle). |

### DnnImageEncoderNode (`config/dnn_img_enc_params.yaml`)

This upstream node (from `isaac_ros_dnn_image_encoder`) is configured here for convenience.

| Parameter | Default | Description |
|---|---|---|
| `input_image_width/height` | `640×480` | Actual camera frame dimensions |
| `network_image_width/height` | `640×640` | Model input resolution — encoder resizes to this |
| `image_mean` / `image_stddev` | `[0.5,0.5,0.5]` | Per-channel normalisation |
| `enable_padding` | `true` | Pad to preserve aspect ratio |
| `keep_aspect_ratio` | `true` | Letterbox instead of stretch |
| `final_tensor_name` | `"input_tensor"` | **Must match** `engine.input_tensor_name` in `obj_det_params.yaml` |
| `image_input_topic` | `arducam/left/nitros_image_rgb8` | Input image topic |
| `tensor_output_topic` | `/dnn_enc/encoded_tensor` | **Must match** `topics.enc_tensor_sub.topic_name` |

---

## Launch

### Full 4-node pipeline (recommended)

Starts `ArducamB0573Node` → `DnnImageEncoderNode` → `ObjectDetectionNode` →
`AnnotatedImagePublisherNode` all in one `component_container_mt`.

```bash
ros2 launch isaac_ros_object_detection composible_node_container.launch.py
```

All four nodes share the same multi-threaded executor. NITROS zero-copy keeps tensors GPU-resident
across the `DnnImageEncoder` → `ObjectDetectionNode` boundary with no serialisation.

### Annotation node only (e.g., when detector runs in a separate container)

```bash
ros2 launch isaac_ros_object_detection annotated_image_publisher.launch.py
```

### Development / debug

```bash
ros2 launch isaac_ros_object_detection dev.launch.py
```

### Verify output

```bash
# Check detections
ros2 topic echo /object_detection/detections

# Check annotated preview (compressed)
ros2 topic hz /object_detection/annotated_image/compressed

# View in RViz2
# Add → By topic → /object_detection/annotated_image/compressed → Camera
```

---

## Tuning Guide

**Reducing false detections**
- Raise `filters.confidence_threshold` (e.g., `0.4`–`0.6`). Note: the annotator has its own
  independent `confidence_threshold` and can be set higher to keep the preview clean.
- Narrow `filters.classes` to only the classes you care about (e.g., `[0]` for person-only).

**DnnImageEncoder input size mismatch**
- `input_image_width/height` in `dnn_img_enc_params.yaml` must match the **actual camera output
  resolution**, not the model size. The encoder resizes internally. Getting this wrong causes
  incorrect aspect-ratio calculation and distorted boxes.

**Tensor name mismatch (common error)**
- If the node prints `GetNamedTensor(...) failed`, the `engine.input_tensor_name` in
  `obj_det_params.yaml` does not match `final_tensor_name` in `dnn_img_enc_params.yaml`. Run
  `inspect_model.py` and verify both match.

**Bounding boxes appear at wrong scale**
- `detection_model_input_width/height` in `annotated_img_publisher_params.yaml` must match
  `engine.input_width/height` in `obj_det_params.yaml`. The annotator uses these values to scale
  boxes from 640×640 model space into the original camera resolution.

**Thread safety / TRT context crash**
- The `ObjectDetectionNode` callback acquires `std::mutex inference_mutex_` on every call. If you
  see crashes under `component_container_mt`, ensure you have not disabled the mutex guard. TRT
  execution contexts are **not** thread-safe — the mutex is mandatory.

**Choosing `"vic"` vs `"cuda"` decoder**
- Only set `hardware_decoder: "vic"` if your camera node publishes a raw `NvBufSurface*` pointer
  (e.g., direct ArgusCamera path without `DnnImageEncoder`). For every other camera source go
  through `DnnImageEncoderNode` and set `hardware_decoder: "cuda"` — it is simpler, faster, and
  does not require a VIC surface allocation.

---

## License

MIT © 2026 WM Nipun Dhananjaya — see [LICENSE](LICENSE) for the full text.