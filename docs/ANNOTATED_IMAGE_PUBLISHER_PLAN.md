# AnnotatedImagePublisherNode — Implementation Plan

> Date: 2026-02-28
> Package: `isaac_ros_object_detection`
> Node: `nvidia::isaac_ros::object_detection::AnnotatedImagePublisherNode`

---

## Goal

Publish annotated images (raw or JPEG-compressed) with bounding boxes and text labels overlaid,
hardware-accelerated on Jetson Orin. Re-publish `CameraInfo` alongside the annotated frame so
RViz2 can display it without manual topic remapping.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Detection sync | Cache pattern (no `message_filters`) | Lower annotation latency; decoupled queues |
| NITROS → BGR | VIC auto-detect on first frame; CUDA swap-kernel fallback | VIC is free HW; fallback is safe |
| VIC output format | RGBA (packed device surface) | VIC can output RGBA; single CUDA kernel strips alpha to BGR |
| Text label rendering | GPU glyph atlas | Atlas built once at startup via OpenCV CPU (`cv::putText`); stamped every frame 100% on GPU |
| JPEG encode | nvJPEG (GPU) guarded by `WITH_NVJPEG=1`; fallback `cv::imencode` | Keeps encode off CPU on Orin |
| JPEG decode (compressed input) | nvJPEG GPU decode → planar BGR planes → CUDA pack to packed BGR | Same |
| CameraInfo re-publish | Fixed at `<annotated_topic>/camera_info` | RViz2 Image display convention |
| Host memory | `cudaHostAlloc` pinned buffers | Avoids pageable-memory copy penalty |
| CUDA stream | `cudaStreamNonBlocking` | CPU returns immediately; one `cudaStreamSynchronize` per frame |

---

## Key Parameters (from `annotated_img_publihser_params.yaml`)

```yaml
sub_topic_image.transport:        "raw" | "compressed" | "nitros"
sub_topic_image.topic_name:       (e.g.) /arducam/left/nitros_image_rgb8
sub_topic_camera_info.topic_name: /arducam/left/image/camera_info
sub_topic_detections.topic_name:  /object_detection/detections
pub_topic_image.topic_name:       /object_detection/annotated_image
pub_topic_image.transport:        "raw" | "compressed"
detection_model_input_width:      640
detection_model_input_height:     640
num_classes:                      80
draw_labels:       true
draw_confidence:   true
confidence_threshold: 0.5
font_size:     0.5
font_thickness: 1
box_thickness:  2
box_color:      [0, 255, 0]   # BGR; [-1,-1,-1] = per-class colors
```

CameraInfo is always re-published at: `<pub_topic_image.topic_name>/camera_info`

---

## Files Created / Modified

### New Files

| File | Purpose |
|---|---|
| `include/isaac_ros_object_detection/annotated_image_publisher_node.hpp` | Node class declaration |
| `src/annotated_image_publisher_node.cpp` | Full node implementation |
| `src/annotation_kernels.cu` | CUDA: bbox draw, fill rects, glyph atlas text stamp, rgb↔bgr helpers |
| `src/annotated_image_publisher_main.cpp` | Standalone entry point (`ros2 run`) |

### Modified Files

| File | Change |
|---|---|
| `include/isaac_ros_object_detection/annotation_kernels.hpp` | Add `annotation_draw_text_gpu()`, `cuda_rgb_to_bgr_inplace()`, `cuda_rgba_to_bgr_packed()`, `cuda_planar_to_packed_bgr()` declarations |
| `src/cuda_kernels.cu` | Add `cuda_rgb_to_bgr_inplace`, `cuda_rgba_to_bgr_packed`, `cuda_planar_to_packed_bgr` helper kernel entry points |
| `package.xml` | Add `<depend>isaac_ros_nitros_image_type</depend>` |
| `launch/composible_node_container.launch.py` | Add `AnnotatedImagePublisherNode` composable entry |
| `launch/` | New standalone `annotated_image_publisher.launch.py` |

---

## Data Flow per Transport Mode

### NITROS input path
```
NitrosImageView (GPU, RGB8)
  → [first frame] probe NvBufSurface handle
      → NvBufSurfTransform (VIC HW: RGB→RGBA, aligned surface)
          → cuda_rgba_to_bgr_packed (CUDA: strips alpha, packed BGR)
      → fallback: cuda_rgb_to_bgr_inplace (CUDA: in-place R↔B swap)
  → d_bgr_device
  → annotation pipeline (below)
```

### Compressed input path
```
sensor_msgs/CompressedImage (JPEG bytes, CPU)
  → [WITH_NVJPEG] nvjpegDecode (GPU) → 3 planar device channels
      → cuda_planar_to_packed_bgr (CUDA: interleave to packed BGR)
  → [fallback] cv::imdecode (CPU) + cudaMemcpyAsync HtoD
  → d_bgr_device
  → annotation pipeline (below)
```

### Raw input path
```
sensor_msgs/Image (CPU, bgr8|rgb8|mono8)
  → cudaMemcpyAsync HtoD (pinned staging)
  → [rgb8] cuda_rgb_to_bgr_inplace (CUDA)
  → d_bgr_device
  → annotation pipeline (below)
```

### Annotation Pipeline (GPU, runs after all input paths)
```
d_bgr_device
  → BBoxDraw[] build (CPU, lightweight: scale + clamp + filter)
  → annotation_draw_rects_gpu  (CUDA: border pixels)
  → [draw_labels=true]
      → pack label chars + positions → device
      → annotation_draw_text_gpu  (CUDA: glyph atlas stamp)
  → cudaStreamSynchronize
  → [COMPRESSED + nvJPEG] nvjpegEncodeImage → nvjpegEncodeRetrieveBitstreamDevice
                           + DtoH → CompressedImage::data → publish
  → [COMPRESSED + fallback] DtoH → cv::imencode → publish
  → [RAW] DtoH via pinned staging → Image (bgr8) → publish
  → CameraInfo (cached, stamp updated) → publish at <topic>/camera_info
```

---

## BBox Coordinate Scaling

Detections from `ObjectDetectionNode` are published in **model-input pixel space** (0..640).
The annotator scales them to the actual image resolution:

```
scale_x = image_width  / model_input_width_  (e.g. 1280/640 = 2.0)
scale_y = image_height / model_input_height_ (e.g.  480/640 = 0.75)

x1 = (center_x - size_x/2) * scale_x
y1 = (center_y - size_y/2) * scale_y
x2 = (center_x + size_x/2) * scale_x
y2 = (center_y + size_y/2) * scale_y
```

---

## Glyph Atlas Layout

```
Atlas: 2D uint8 texture,  width = 95 * glyph_w,  height = glyph_h
Glyph index = char - 32  (covers ASCII 32..126, i.e. space through tilde)
Glyph pixel (row r, col c) at atlas_x = glyph_idx * glyph_w + c,  atlas_y = r
```
Built once at node startup using `cv::putText` with the configured `font_size` and `font_thickness`.

---

## Kernel Signatures (annotation_kernels.hpp)

```cpp
// BBoxDraw struct: {int x1,y1,x2,y2; uint8_t b,g,r; int thickness}
bool annotation_draw_rects_gpu(uint8_t* d_img, int w, int h, int pitch,
    const BBoxDraw* h_boxes, int n, cudaStream_t stream);

bool annotation_draw_fill_rects_gpu(uint8_t* d_img, int w, int h, int pitch,
    const BBoxDraw* h_boxes, int n, cudaStream_t stream);

// GPU text stamp via glyph atlas
bool annotation_draw_text_gpu(uint8_t* d_img, int img_w, int img_h, int pitch,
    const uint8_t* d_labels_buf, const int* d_offsets, const int* d_lens,
    const int* d_x, const int* d_y,
    uint8_t text_b, uint8_t text_g, uint8_t text_r,
    int num_labels,
    cudaTextureObject_t atlas_tex, int glyph_w, int glyph_h,
    cudaStream_t stream);

// Color-format helpers (packed BGR device buffer)
void cuda_rgb_to_bgr_inplace(uint8_t* d_bgr, int w, int h, cudaStream_t stream);
void cuda_rgba_to_bgr_packed(const uint8_t* d_rgba, uint8_t* d_bgr, int w, int h, cudaStream_t stream);
void cuda_planar_to_packed_bgr(const uint8_t* d_b, const uint8_t* d_g, const uint8_t* d_r,
    uint8_t* d_bgr, int w, int h, cudaStream_t stream);
```

---

## VIC Auto-Detection Logic (NITROS path)

On the **first NITROS frame**:
1. Call `NvBufSurfaceFromFd(-1, &test_surf, &params)` — if the NITROS backing handle  
   comes from `nvbufsurface` APIs, it will have a valid DMA-buf FD.
2. More reliable: attempt `NvBufSurfTransformAsync(src_surf, vic_dst_, params, stream)` —  
   VIC returns `NvBufSurfTransformError_Success` only if src is a valid NvBufSurface.
3. On success → `use_vic_ = true`. On failure → `use_vic_ = false`, log warning, use CUDA fallback.

In practice the NITROS RGB8 buffer from `ArducamB0573Node` (which uses GStreamer nvvidconv → TypeAdapter) 
**is NOT an NvBufSurface** — the TypeAdapter allocates a plain `cudaMalloc` buffer. So the fallback 
`cuda_rgb_to_bgr_inplace` kernel will always be used for this camera. VIC is kept ready for future 
camera nodes that do use NvBufSurface.

---

## Build

```bash
cd /home/orin/ros2_ws
colcon build --packages-select isaac_ros_object_detection \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Test (standalone)

```bash
# With /arducam and /object_detection/detections already running:
ros2 launch isaac_ros_object_detection annotated_image_publisher.launch.py
# or:
ros2 run isaac_ros_object_detection annotated_image_publisher_node \
  --ros-args --params-file \
  ~/ros2_ws/src/ros2_px4_object_follower/isaac_ros_object_detection/config/annotated_img_publihser_params.yaml

# RViz2: add Image display → /object_detection/annotated_image
#  CameraInfo auto-found at    /object_detection/annotated_image/camera_info

ros2 topic hz /object_detection/annotated_image   # should match camera FPS
ros2 topic info /object_detection/annotated_image -v
```
