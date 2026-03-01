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
  // yolo26n has NMS embedded: output is (1, 300, 6)  float32  [x1,y1,x2,y2,conf,cls_id]
  const size_t out_bytes = 1 * 300 * 6 * sizeof(float);

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
  auto objects = DecodeOutputGPU(d_output_, 300);

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
// Post-processing: GPU decode + CPU NMS
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Object>
ObjectDetectionNode::DecodeOutputGPU(
  const void* d_output,
  int /*num_predictions*/  // 300 for yolo26n (post-NMS embedded in model)
)
{
  // yolo26n engine has NMS baked in. Output: (1, 300, 6) float32
  // Each row: [x1, y1, x2, y2, confidence, class_id]  (xyxy absolute pixels at 640x640)
  // Rows with confidence == 0 are padding — filter them out.
  constexpr int MAX_DETS = 300;
  constexpr int FIELDS   = 6;

  std::vector<float> h_raw(MAX_DETS * FIELDS, 0.0f);

  // Sync inference before reading results
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
  CUDA_CHECK(cudaMemcpy(h_raw.data(), d_output,
                        MAX_DETS * FIELDS * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::vector<Object> out;
  out.reserve(MAX_DETS);

  for (int i = 0; i < MAX_DETS; ++i) {
    const float* b = h_raw.data() + i * FIELDS;

    const float conf = b[4];

    // Skip zero-confidence padding rows and low-confidence detections
    if (conf < confidence_threshold_) continue;

    // yolo26n output is xyxy (absolute pixels at input resolution).
    // byte_track::Object requires center-format {cx, cy, w, h}.
    // Clamp coordinates to image bounds and skip degenerate boxes (can occur
    // when the camera moves fast and objects are partially off-screen).
    const float x1 = std::max(0.f, std::min(b[0], static_cast<float>(input_width_)));
    const float y1 = std::max(0.f, std::min(b[1], static_cast<float>(input_height_)));
    const float x2 = std::max(0.f, std::min(b[2], static_cast<float>(input_width_)));
    const float y2 = std::max(0.f, std::min(b[3], static_cast<float>(input_height_)));

    const float w = x2 - x1;
    const float h = y2 - y1;
    // Skip zero/negative-size boxes — they produce NaN in TlwhToXyah (div-by-h)
    // which then poisons the LAPJV cost matrix and causes a segfault.
    if (w <= 0.f || h <= 0.f) continue;

    Object obj;
    obj.x     = (x1 + x2) * 0.5f;        // center x
    obj.y     = (y1 + y2) * 0.5f;        // center y
    obj.w     = w;
    obj.h     = h;
    obj.prob  = conf;
    obj.label = static_cast<int>(b[5]);

    // Apply class filter (empty allowed_classes_ means keep all)
    if (!allowed_classes_.empty() && allowed_classes_.find(obj.label) == allowed_classes_.end())
      continue;

    out.push_back(obj);
  }

  // NMS already done inside the model.
  return out;
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





