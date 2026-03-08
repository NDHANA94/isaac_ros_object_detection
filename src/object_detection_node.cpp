/** 
 * ─────────────────────────────────────────────────────────────────────────────
 * MIT License

 * Copyright (c) 2026 WM Nipun Dhananjaya

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * ───────────────────────────────────────────────────────────────────────────── 
*/

// @file: object_detection_node.cpp
// @brief: Implementation of the ObjectDetectionNode composable ROS2 node.
// @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
// @company: Sintez.LLC
// @date: 2026-03-01

/** ────────────────────────────────────────────────────────────────────────────
 * @file: object_detection_node.cpp
 * @brief: Implementation of the ObjectDetectionNode composable ROS2 node.
 * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
 * @company: Sintez.LLC
 * @date: 2026-03-01
 * 
 * This file implements the ObjectDetectionNode, which is a ROS2 composable node that
 * subscribes to a NITROS image topic, runs a TensorRT object detection model on
 * the GPU, and publishes the resulting detections as a Detection2DArray.
 * 
 * Pipeline block diagram: TODO
 * 
 * ─────────────────────────────────────────────────────────────────────────────
*/

#include "isaac_ros_object_detection/object_detection_node.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <sstream>

#include "std_msgs/msg/header.hpp"
#include "vision_msgs/msg/bounding_box2_d.hpp"
#include "visualization_msgs/msg/marker.hpp"


// CUDA helpers
#define CUDA_CHECK(call)                                                                    \
  do {                                                                                      \
      cudaError_t err = (call);                                                             \
      if (err != cudaSuccess) {                                                             \
          throw std::runtime_error(                                                         \
            std::string("[CUDA] ") + cudaGetErrorString(err) +                              \
            " at " __FILE__ ":" + std::to_string(__LINE__));                                \
      }                                                                                     \
  } while (0)



// --- External CUDA kernals (implemented in cuda_kernels.cu) ---
extern "C" void cuda_preprocess_nv12(
  const void * d_src,   int src_w,  int src_h,
  float * d_dst,        int dst_w,  int dst_h,
  cudaStream_t stream);

extern "C" void cuda_normalize_rgba(
  const void* d_rgba, float* d_dst,
  int w, int h,
  cudaStream_t stream);
// cuda_decode_yolo_output removed: yolo26n has NMS embedded, output is already (1,300,6)

extern "C" void cuda_decode_yolo_output(
  const float* d_raw, int num_predictions, int num_classes,
  float conf_thresh, float* d_boxes_out, int* d_count_out, cudaStream_t stream);






namespace nvidia
{
namespace isaac_ros
{
namespace object_detection
{

// ─────────────────────────────────────────────────────────────────────────────
// Construction / Destruction
// ─────────────────────────────────────────────────────────────────────────────

ObjectDetectionNode::ObjectDetectionNode(const rclcpp::NodeOptions & options)
: Node("object_detection_node", options)
{
  // --- declare parameters ---

  // Hardware + misc
  hardware_decoder_           = this->declare_parameter<std::string>("hardware_decoder",       "auto");

  // Engine group
  model_path_                 = this->declare_parameter<std::string>("engine.model_path",              "");
  class_names_path_           = this->declare_parameter<std::string>("engine.class_names_path",        "");
  quantization_               = this->declare_parameter<std::string>("engine.quantization",            "fp16");
  input_tensor_name_          = this->declare_parameter<std::string>("engine.input_tensor_name",       "input_tensor");
  trt_input_binding_name_     = this->declare_parameter<std::string>("engine.trt_input_binding_name",  "images");
  output_tensor_name_         = this->declare_parameter<std::string>("engine.output_tensor_name",      "output0");
  // input_width_ / input_height_ are auto-detected from the TRT engine after BuildEngine()
  num_classes_                = this->declare_parameter("engine.num_classes",   80);

  // Filters group
  confidence_threshold_       = static_cast<float>(this->declare_parameter("filters.confidence_threshold", 0.20));
  nms_threshold_              = static_cast<float>(this->declare_parameter("filters.nms_threshold",        0.45));
  allowed_classes_raw_        = this->declare_parameter<std::vector<int64_t>>("filters.classes", std::vector<int64_t>{});
  for (int64_t c : allowed_classes_raw_) if(c != -1) {allowed_classes_.insert(static_cast<int>(c));}

  // Topics group
  sub_topic_          = this->declare_parameter<std::string>("topics.enc_tensor_sub.topic_name",    "/arducam/left/nitros_image");
  sub_reliability_    = this->declare_parameter<std::string>("topics.enc_tensor_sub.qos.reliability", "best_effort");
  sub_durability_     = this->declare_parameter<std::string>("topics.enc_tensor_sub.qos.durability",  "volatile");
  sub_depth_          = this->declare_parameter("topics.enc_tensor_sub.qos.depth",                   10);
  pub_topic_          = this->declare_parameter<std::string>("topics.detection_pub.topic_name",      "/detections");
  pub_reliability_    = this->declare_parameter<std::string>("topics.detection_pub.qos.reliability",  "best_effort");
  pub_durability_     = this->declare_parameter<std::string>("topics.detection_pub.qos.durability",   "volatile");
  pub_depth_          = this->declare_parameter("topics.detection_pub.qos.depth",                    10);



  if(model_path_.empty()) {
    RCLCPP_FATAL(get_logger(), "Parameter 'engine.model_path' is required but not set. Shutting down.");
    rclcpp::shutdown();
    return;
  }

  // --- Load class names (optional) ---
  LoadClassNames();


  // --- CUDA Setup -----------------------------------------------------------------------------------
  // Non-blocking stream: CPU returns immediately after queuing work on GPU, 
  // synchronization is handled via CUDA events
  CUDA_CHECK(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));

  // --- TensorRT engine -------------------------------------------------------------------------------
  // BuildEngine() auto-detects input_width_ / input_height_ from the engine's binding shape,
  // so it must run before any buffer allocation that depends on those dimensions.
  BuildEngine();

  // Device-side I/O buffers (persistent - allocated once, reused every frame).
  // TRT engine I/O tensors are always float32 regardless of FP16/INT8 precision setting
  // (FP16/INT8 only affects internal layer computation, not the I/O buffer format).
  const size_t in_bytes  = 1 * 3 * input_width_ * input_height_ * sizeof(float);  // [1, 3, H, W]
  // Output buffer sized from the auto-detected engine shape (set by BuildEngine()).
  //   NMS-embedded (yolo26n): out_rows_=300, out_cols_=6   → 1 800 floats
  //   YOLOv8 native (yolov8n): out_rows_=84,  out_cols_=8400 → 705 600 floats
  const size_t out_bytes = static_cast<size_t>(out_rows_) * static_cast<size_t>(out_cols_) * sizeof(float);

  // Allocate device buffers
  CUDA_CHECK(cudaMalloc(&d_input_, in_bytes));
  CUDA_CHECK(cudaMalloc(&d_output_, out_bytes));

  // Bind device buffers to the TRT execution context now that they are allocated
  trt_context_->setTensorAddress(trt_input_binding_name_.c_str(), d_input_);
  trt_context_->setTensorAddress(output_tensor_name_.c_str(), d_output_);

  // --- VIC workspace ---------------------------------------------------------------------------------
  if (hardware_decoder_ != "cuda")  AllocVICSurface();

  
  // --- NITROS subscriber ----------------------------------------------------------------------------
  // ManagedNitrosSubscriber keeps tensor on the GPU -- no host copies.
  // The upstream pipeline (ArgusCamera -> DNN Image Encoder) must publish
  // a NitrosTensorList on this topic.
  nitros_sub_ = 
    std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>>(
        this, 
        sub_topic_, 
        nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
        std::bind(&ObjectDetectionNode::InputCallback, this, std::placeholders::_1),
        nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{},
        BuildQoS(sub_reliability_, sub_durability_, sub_depth_)
    );

  // --- Publishers ------------------------------------------------------------------------------------
  detection_pub_ = 
    create_publisher<vision_msgs::msg::Detection2DArray>(
      pub_topic_, 
      BuildQoS(pub_reliability_, pub_durability_, pub_depth_)
    );


  RCLCPP_INFO(get_logger(), 
    "ObjectDetectionNode ready.\n"
    "  hardware_decoder=%s  quantization=%s\n"
    "  model=%s\n"
    "  classes_filtered=%zu  conf_thresh=%.2f  nms_thresh=%.2f\n"
    "  sub=%s  pub=%s\n",
    hardware_decoder_.c_str(), quantization_.c_str(),
    model_path_.c_str(),
    allowed_classes_.size(), confidence_threshold_, nms_threshold_,
    sub_topic_.c_str(), pub_topic_.c_str());

}


ObjectDetectionNode::~ObjectDetectionNode()
{
  if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
  if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
  if (cuda_stream_) { cudaStreamDestroy(cuda_stream_); cuda_stream_ = nullptr; }
  if (vic_dst_surf_) { NvBufSurfaceDestroy(vic_dst_surf_); vic_dst_surf_ = nullptr; }
}


// ─────────────────────────────────────────────────────────────────────────────
// TensorRT -- Build / Deserialize Engine
// ─────────────────────────────────────────────────────────────────────────────

void 
ObjectDetectionNode::BuildEngine()
{
  std::ifstream file(model_path_, std::ios::binary | std::ios::ate);
  if(!file) throw std::runtime_error("Cannot open TRT engine: " + model_path_);

  const size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buf(size);
  file.read(buf.data(), size);

  trt_runtime_.reset(nvinfer1::createInferRuntime(trt_logger_));
  trt_engine_.reset(trt_runtime_->deserializeCudaEngine(buf.data(), size));
  if(!trt_engine_) throw std::runtime_error("Failed to deserialize TRT engine.");

  trt_context_.reset(trt_engine_->createExecutionContext());
  if(!trt_context_) throw std::runtime_error("Failed to create TRT execution context.");

  // ── Auto-detect input dimensions from the engine binding shape ──────────────
  // Engine input tensor is [1, 3, H, W]; indices 2 and 3 give H and W.
  auto dims = trt_engine_->getTensorShape(trt_input_binding_name_.c_str());
  if (dims.nbDims == 4 && dims.d[2] > 0 && dims.d[3] > 0) {
    input_height_ = static_cast<int>(dims.d[2]);
    input_width_  = static_cast<int>(dims.d[3]);
    RCLCPP_INFO(get_logger(),
      "Auto-detected model input dimensions from TRT engine: %dx%d",
      input_width_, input_height_);
  } else {
    RCLCPP_WARN(get_logger(),
      "Could not auto-detect input dimensions from TRT engine "
      "(binding '%s', nbDims=%d). Keeping defaults %dx%d.",
      trt_input_binding_name_.c_str(), dims.nbDims, input_width_, input_height_);
  }

  // ── Auto-detect output tensor shape to select decode path ─────────────────────────────
  // • NMS-embedded (yolo26n):  (1, 300,  6) → out_rows_=300, out_cols_=6
  //   Each row: [x1, y1, x2, y2, conf, cls_id]  (no CPU NMS needed)
  // • YOLOv8 native (yolov8n): (1,  84, 8400) → out_rows_=84, out_cols_=8400
  //   Row k col j: raw[k*8400+j]; rows 0-3 = cx/cy/w/h, rows 4-83 = class scores
  //   (requires CPU NMS)
  auto out_dims = trt_engine_->getTensorShape(output_tensor_name_.c_str());
  if (out_dims.nbDims == 3 && out_dims.d[1] > 0 && out_dims.d[2] > 0) {
    out_rows_ = static_cast<int>(out_dims.d[1]);
    out_cols_ = static_cast<int>(out_dims.d[2]);
    // Heuristic: YOLOv8 has many more anchors (cols) than attribute rows.
    // yolo26n is the opposite (300 rows, 6 cols).
    is_yolov8_format_ = (out_cols_ > out_rows_);
    RCLCPP_INFO(get_logger(),
      "Auto-detected output shape: (1, %d, %d) — %s format",
      out_rows_, out_cols_,
      is_yolov8_format_ ? "YOLOv8 transposed (CPU NMS)" : "NMS-embedded xyxy+conf+cls");
  } else {
    RCLCPP_WARN(get_logger(),
      "Could not auto-detect output shape from TRT engine "
      "(binding '%s', nbDims=%d). Keeping defaults %dx%d.",
      output_tensor_name_.c_str(), out_dims.nbDims, out_rows_, out_cols_);
  }

  // Bind device buffers to named I/O tensors.
  // NOTE: d_input_ / d_output_ are allocated in the constructor after BuildEngine() returns;
  //       setTensorAddress() is called there once the buffers exist.

  RCLCPP_INFO(get_logger(), "Successfully loaded TRT engine from %s", model_path_.c_str());
}


// ─────────────────────────────────────────────────────────────────────────────
// VIC surface allocation
// ─────────────────────────────────────────────────────────────────────────────

void
ObjectDetectionNode::AllocVICSurface()
{
  NvBufSurfaceCreateParams cp{};
  cp.gpuId    = 0;
  cp.width    = input_width_;
  cp.height   = input_height_;
  cp.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
  cp.layout   = NVBUF_LAYOUT_PITCH;
  cp.memType  = NVBUF_MEM_CUDA_DEVICE;

  if (NvBufSurfaceCreate(&vic_dst_surf_, 1, &cp) != 0) 
    throw std::runtime_error("Failed to allocate VIC destination NvBufSurface.");
  
}


// ─────────────────────────────────────────────────────────────────────────────
// Main callback -- NitrosTensorListView arrives GPU-resident
// ─────────────────────────────────────────────────────────────────────────────

void
ObjectDetectionNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorListView& msg)
{
  // Serialise inference: component_container_mt is multi-threaded; TRT context,
  // CUDA stream, and d_input_/d_output_ are not thread-safe.
  std::lock_guard<std::mutex> lock(inference_mutex_);

  // Build a ROS header from the NITROS timestamp.
  // NitrosTensorListView has no GetHeader() — use GetTimestampSeconds / Nanoseconds.
  std_msgs::msg::Header header;
  header.stamp.sec     = msg.GetTimestampSeconds();
  header.stamp.nanosec = msg.GetTimestampNanoseconds();

  // GetNamedTensor returns by value and throws std::runtime_error when the
  // name is not in the list.  Catch it here so a misconfigured tensor name
  // produces an actionable ERROR instead of a silent callback crash.
  const void* d_src = nullptr;
  try {
    // Extract the GPU buffer pointer immediately; the returned NitrosTensorView
    // is a temporary but the underlying GXF pool buffer remains valid for the
    // lifetime of this callback.
    d_src = msg.GetNamedTensor(input_tensor_name_).GetBuffer();
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
      "GetNamedTensor('%s') failed: %s  "
      "-- check that 'input_tensor_name' in params.yaml matches the "
      "DNN encoder's 'final_tensor_name' (default: 'input_tensor').",
      input_tensor_name_.c_str(), e.what());
    return;
  }

  // ── Pre-process ───────────────────────────────────────────────
  if (hardware_decoder_ == "vic" || hardware_decoder_ == "auto") {
    // VIC path: upstream feeds a raw YUV/NV12 NvBufSurface (eg. ArgusCamera direct path)
    NvBufSurface* src_surf = const_cast<NvBufSurface*>(
      reinterpret_cast<const NvBufSurface*>(d_src)
    );
    PreprocessVIC(src_surf, d_input_);
  } else {
    // CUDA path: input tensor is float32 NCHW from NITROS/DnnImageEncoder.
    const size_t in_bytes = 1 * 3 * input_width_ * input_height_ * sizeof(float);
    cudaError_t ce = cudaMemcpy(d_input_, d_src, in_bytes, cudaMemcpyDeviceToDevice);
    if (ce != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "cudaMemcpy failed: %s", cudaGetErrorString(ce));
      return;
    }
  }

  // ── TRT inference (async, GPU only) ─────────────────────────────
  RunInference();

  // ── Decode detections on GPU ────────────────────────────────────
  auto objects = DecodeOutputGPU(d_output_);

  // Debug: ros log detected object list
  RCLCPP_DEBUG(get_logger(), "Detected %zu objects", objects.size());
  for (size_t i = 0; i < objects.size(); ++i) {
    const auto& obj = objects[i];
    RCLCPP_DEBUG(get_logger(), "  [%zu] class=%d  conf=%.2f  x=%.1f y=%.1f w=%.1f h=%.1f",
      i, obj.label, obj.prob, obj.x, obj.y, obj.w, obj.h);
  }

  // Publish detections as ROS messages
  PublishDetections(objects, header);

}



// ─────────────────────────────────────────────────────────────────────────────
// Pre-processing: VIC (hardware-accelerated, zero-copy GPU) with CUDA fallback
// ─────────────────────────────────────────────────────────────────────────────

void
ObjectDetectionNode::PreprocessVIC(NvBufSurface* src_surf, void* d_dst_float)
{
  // NvBufSurfTransform runs entirely on the VIC hardware engine:
  //    - YUV(NV12) -> RGBA color conversion
  //    - Bilinear resize to input_width_ * input_height_
  //  Zero CPU involvement, no host copies - the output is a GPU pointer ready for inference.

  NvBufSurfTransformParams tp{};
  tp.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
  tp.transform_filter = NvBufSurfTransformInter_Bilinear;

  if(NvBufSurfTransform(src_surf, vic_dst_surf_, &tp) != NvBufSurfTransformError_Success) {
    RCLCPP_ERROR(get_logger(), "VIC NvBufSurfTransform failed -- Skipping frame.");
    return;
  }

  // Normalize RGBA [0, 255] -> float [0, 1] and convert to NCHW via CUDA kernel.
  // This is a trivial element-wise op -- executes in < 0.1 ms on GPU, no host copy needed.
  cuda_normalize_rgba(
    vic_dst_surf_->surfaceList[0].dataPtr,
    static_cast<float*>(d_dst_float),
    input_width_, input_height_,
    cuda_stream_
  );
}



// ─────────────────────────────────────────────────────────────────────────────
// Pre-processing: CUDA fallback path (when VIC is disabled )
// ─────────────────────────────────────────────────────────────────────────────
void
ObjectDetectionNode::PreprocessCUDA(
  const void* d_src_nv12,
  int src_w,
  int src_h,
  void* d_dst_float)
{
  cuda_preprocess_nv12(
    d_src_nv12, src_w, src_h,
    static_cast<float*>(d_dst_float), input_width_, input_height_,
    cuda_stream_
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// Inference
// ─────────────────────────────────────────────────────────────────────────────

void
ObjectDetectionNode::RunInference()
{
  // enqueueV3: fully asynchronous - CPU returns immediately.
  // The CUDA stream serializes VIC output -> TRT kernel.
  if(!trt_context_->enqueueV3(cuda_stream_)) RCLCPP_ERROR(get_logger(), "TRT enqueueV3 failed: Failed to enqueue TRT inference.");
}



// ─────────────────────────────────────────────────────────────────────────────
// Post-processing helpers (file-scope, not exported)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// IoU of two center-format (cx, cy, w, h) boxes.
float iou_center(const Object& a, const Object& b)
{
  const float ax1 = a.x - a.w * 0.5f, ay1 = a.y - a.h * 0.5f;
  const float ax2 = a.x + a.w * 0.5f, ay2 = a.y + a.h * 0.5f;
  const float bx1 = b.x - b.w * 0.5f, by1 = b.y - b.h * 0.5f;
  const float bx2 = b.x + b.w * 0.5f, by2 = b.y + b.h * 0.5f;
  const float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
  const float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
  const float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
  if (inter <= 0.f) return 0.f;
  return inter / (a.w * a.h + b.w * b.h - inter);
}

// Greedy per-class NMS. `dets` is sorted descending by prob on return.
std::vector<Object> greedy_nms(std::vector<Object> dets, float iou_thresh)
{
  std::sort(dets.begin(), dets.end(),
            [](const Object& a, const Object& b) { return a.prob > b.prob; });
  std::vector<bool> keep(dets.size(), true);
  for (size_t i = 0; i < dets.size(); ++i) {
    if (!keep[i]) continue;
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (keep[j] && dets[i].label == dets[j].label &&
          iou_center(dets[i], dets[j]) > iou_thresh) {
        keep[j] = false;
      }
    }
  }
  std::vector<Object> out;
  out.reserve(dets.size());
  for (size_t i = 0; i < dets.size(); ++i) {
    if (keep[i]) out.push_back(dets[i]);
  }
  return out;
}

} // anonymous namespace


// ─────────────────────────────────────────────────────────────────────────────
// Post-processing: decode TRT output → Object list
//
// Supports two output layouts determined at runtime from the engine shape:
//
//  NMS-embedded  (e.g. yolo26n): (1, N, 6)
//    Each row i: [x1, y1, x2, y2, conf, cls_id]  (xyxy absolute pixels, post-NMS)
//    is_yolov8_format_ == false
//
//  YOLOv8 native (e.g. yolov8n): (1, 4+C, A)
//    Memory layout: row-major, row k has A values for all anchors.
//    row 0..3 = cx/cy/w/h; rows 4..4+C-1 = per-class scores.
//    conf = max class score (no separate objectness in YOLOv8).
//    is_yolov8_format_ == true  →  CPU NMS applied after decode.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Object>
ObjectDetectionNode::DecodeOutputGPU(const void* d_output)
{
  // Copy the whole output tensor to host in one DtoH transfer.
  const size_t total_floats = static_cast<size_t>(out_rows_) * static_cast<size_t>(out_cols_);
  std::vector<float> h_raw(total_floats, 0.0f);

  CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
  CUDA_CHECK(cudaMemcpy(h_raw.data(), d_output,
                        total_floats * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // ── Branch A: NMS-embedded format (1, MAX_DETS, 6) ───────────────────────
  if (!is_yolov8_format_) {
    const int MAX_DETS = out_rows_;   // e.g. 300
    // out_cols_ must be exactly 6 for this format
    std::vector<Object> out;
    out.reserve(MAX_DETS);

    for (int i = 0; i < MAX_DETS; ++i) {
      const float* b = h_raw.data() + i * out_cols_;
      const float conf = b[4];
      if (conf < confidence_threshold_) continue;

      const float x1 = std::max(0.f, std::min(b[0], static_cast<float>(input_width_)));
      const float y1 = std::max(0.f, std::min(b[1], static_cast<float>(input_height_)));
      const float x2 = std::max(0.f, std::min(b[2], static_cast<float>(input_width_)));
      const float y2 = std::max(0.f, std::min(b[3], static_cast<float>(input_height_)));
      const float w  = x2 - x1;
      const float h  = y2 - y1;
      // Zero/negative boxes poison the Kalman filter — skip them.
      if (w <= 0.f || h <= 0.f) continue;

      Object obj;
      obj.x     = (x1 + x2) * 0.5f;
      obj.y     = (y1 + y2) * 0.5f;
      obj.w     = w;
      obj.h     = h;
      obj.prob  = conf;
      obj.label = static_cast<int>(b[5]);

      if (!allowed_classes_.empty() &&
          allowed_classes_.find(obj.label) == allowed_classes_.end())
        continue;

      out.push_back(obj);
    }
    return out;   // NMS already done inside model
  }

  // ── Branch B: YOLOv8 native format (1, 4+C, A) ───────────────────────────
  // out_rows_ = 4 + num_classes (e.g. 84),  out_cols_ = num_anchors (e.g. 8400)
  const int A       = out_cols_;           // number of anchors
  const int num_cls = out_rows_ - 4;       // number of classes

  std::vector<Object> candidates;
  candidates.reserve(512);

  for (int j = 0; j < A; ++j) {
    // bbox — rows 0-3, column j
    const float cx = h_raw[0 * A + j];
    const float cy = h_raw[1 * A + j];
    const float bw = h_raw[2 * A + j];
    const float bh = h_raw[3 * A + j];
    if (bw <= 0.f || bh <= 0.f) continue;

    // Find the best class score — rows 4..4+num_cls-1, column j
    float best_score = -1.f;
    int   best_cls   = -1;
    for (int k = 0; k < num_cls; ++k) {
      const float s = h_raw[(4 + k) * A + j];
      if (s > best_score) { best_score = s; best_cls = k; }
    }
    if (best_score < confidence_threshold_) continue;

    if (!allowed_classes_.empty() &&
        allowed_classes_.find(best_cls) == allowed_classes_.end())
      continue;

    // Clamp center to image bounds
    const float cx_c = std::max(0.f, std::min(cx, static_cast<float>(input_width_)));
    const float cy_c = std::max(0.f, std::min(cy, static_cast<float>(input_height_)));

    Object obj;
    obj.x     = cx_c;
    obj.y     = cy_c;
    obj.w     = bw;
    obj.h     = bh;
    obj.prob  = best_score;
    obj.label = best_cls;
    candidates.push_back(obj);
  }

  return greedy_nms(std::move(candidates), nms_threshold_);
}









// ─────────────────────────────────────────────────────────────────────────────
// Publishers
// ─────────────────────────────────────────────────────────────────────────────

void
ObjectDetectionNode::PublishDetections(
  const std::vector<Object> & objects,
  const std_msgs::msg::Header& header
)
{
  auto msg = std::make_unique<vision_msgs::msg::Detection2DArray>();
  msg->header = header;

  for (const auto& obj : objects) {
    vision_msgs::msg::Detection2D det;
    det.header = header;
    // Normalise bbox coordinates to [0, 1] relative to the model input size so
    // downstream nodes (annotated image publisher, target locker, etc.) do not
    // need to know the engine's fixed input resolution.
    const float inv_w = 1.0f / static_cast<float>(input_width_);
    const float inv_h = 1.0f / static_cast<float>(input_height_);
    det.bbox.center.position.x = obj.x * inv_w;
    det.bbox.center.position.y = obj.y * inv_h;
    det.bbox.size_x = obj.w * inv_w;
    det.bbox.size_y = obj.h * inv_h;

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    // Use class name if available, otherwise fall back to numeric string
    if (obj.label >= 0 && static_cast<size_t>(obj.label) < class_names_.size()) {
      hyp.hypothesis.class_id = class_names_[obj.label];
    } else {
      hyp.hypothesis.class_id = std::to_string(obj.label);
    }
    hyp.hypothesis.score = obj.prob;
    det.results.push_back(hyp);

    msg->detections.push_back(det);
  }

  detection_pub_->publish(std::move(msg));
}





// ─────────────────────────────────────────────────────────────────────────────
// BuildQoS helper
// ─────────────────────────────────────────────────────────────────────────────

rclcpp::QoS
ObjectDetectionNode::BuildQoS(
  const std::string & reliability,
  const std::string & durability,
  int depth) const
{
  rclcpp::QoS qos(static_cast<size_t>(depth));
  qos.reliability(
    reliability == "best_effort"
      ? RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
      : RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos.durability(
    durability == "transient_local"
      ? RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL
      : RMW_QOS_POLICY_DURABILITY_VOLATILE);
  return qos;
}


// ─────────────────────────────────────────────────────────────────────────────
// LoadClassNames — reads one class name per line from class_names_path_
// ─────────────────────────────────────────────────────────────────────────────

void
ObjectDetectionNode::LoadClassNames()
{
  if (class_names_path_.empty()) {
    RCLCPP_INFO(get_logger(), "No class_names_path set — class IDs will be used as-is.");
    return;
  }
  std::ifstream f(class_names_path_);
  if (!f) {
    RCLCPP_WARN(get_logger(), "Cannot open class names file: %s", class_names_path_.c_str());
    return;
  }
  std::string line;
  while (std::getline(f, line)) {
    // Strip trailing whitespace / CR
    while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
      line.pop_back();
    class_names_.push_back(line);
  }
  RCLCPP_INFO(get_logger(), "Loaded %zu class names from %s",
    class_names_.size(), class_names_path_.c_str());
}


} // namespace object_detection
} // namespace isaac_ros
} // namespace nvidia

// ─────────────────────────────────────────────────────────────────────────────
// rclcpp component registration
// This macro compiles a NodeFactory symbol into the shared library so that
// component_container(_mt) can discover and load ObjectDetectionNode at runtime.
// rclcpp_components_register_node() in CMakeLists only creates the standalone
// executable entry point — it does NOT embed the factory in the .so.
// ─────────────────────────────────────────────────────────────────────────────
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::object_detection::ObjectDetectionNode)





