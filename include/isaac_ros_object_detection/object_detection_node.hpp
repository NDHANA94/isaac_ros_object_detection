#pragma once

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

/** ────────────────────────────────────────────────────────────────────────────
 * @file: object_detection_node.hpp
 * @brief: Header for the ObjectDetectionNode composable ROS2 node.
 * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
 * @company: Sintez.LLC
 * @date: 2026-03-01
 * 
 * This header declares the C++ class for the ObjectDetectionNode, which is a 
 * ROS2 composable node that subscribes to a NITROS image topic, runs a TensorRT 
 * object detection model on the GPU, and publishes the resulting detections 
 * as a Detection2DArray.  
 * 
 * The node is designed to be flexible and configurable via ROS2 parameters, 
 * allowing users to specify the model path, class names, input/output tensor 
 * names, detection thresholds, topic names, and QoS settings.  
 * 
 * The node uses hardware-accelerated pre-processing with the VIC engine when 
 * possible, and falls back to CUDA-based pre-processing if the input format 
 * is already compatible with the model.  The main inference and decoding 
 * steps are performed entirely on the GPU to minimize latency, and the node 
 * is designed to be thread-safe for use in a multi-threaded executor.  
 * The implementation details are in the corresponding .cpp file, while this 
 * header focuses on the class declaration and member variables.  
 * ─────────────────────────────────────────────────────────────────────────────
*/

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// ROS2
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/logging.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

// Isaac ROS NITROS -- zero-copy GPU tensor transport
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

// TensorRT
#include <atomic>
#include <mutex>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

// NvBufSurface (VIC pre/post-processing)
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>



namespace nvidia
{
namespace isaac_ros
{
namespace object_detection
{



// Input detection (from YOLO decoder)
struct Object
{
  float x{};          // center x (pixels)
  float y{};          // center y (pixels)
  float w{};          // width    (pixels)
  float h{};          // height   (pixels)
  float prob{};       // confidence score
  int   label{-1};    // class id
};


// ─────────────────────────────────────────────────────────────────────────────
// Tensor Logger (minimal)
// ─────────────────────────────────────────────────────────────────────────────
class TRTLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING) {
            RCLCPP_WARN(rclcpp::get_logger("TRTLogger"), "%s", msg);
        }
    }
};



// ─────────────────────────────────────────────────────────────────────────────
// NODE
// ─────────────────────────────────────────────────────────────────────────────


class ObjectDetectionNode : public rclcpp::Node
{
public:
    explicit ObjectDetectionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~ObjectDetectionNode();

private:
    // --- Parameters -------------------------------------------------------------

    // Hardware decoder: "auto" | "vic" | "cuda"
    std::string     hardware_decoder_{"auto"};

    // Engine
    std::string     model_path_{};
    std::string     class_names_path_{};
    std::string     quantization_{"fp16"};
    std::string     input_tensor_name_{"input_tensor"};
    std::string     trt_input_binding_name_{"images"};
    std::string     output_tensor_name_{"output0"};

    //
    mutable std::mutex  inference_mutex_;           // serialises TRT context + CUDA buffers

    // Input shape — auto-detected from the TRT engine's binding after BuildEngine().
    // These fallback defaults are only used if the engine query fails.
    int             input_width_{640};
    int             input_height_{640};
    int             num_classes_{80};

    // Output shape — auto-detected from the TRT engine's output binding after BuildEngine().
    // out_rows_ = dim[1], out_cols_ = dim[2]
    //   NMS-embedded  (e.g. yolo26n): (1, 300,  6) → out_rows_=300, out_cols_=6
    //   YOLOv8 native (e.g. yolov8n): (1,  84, 8400) → out_rows_=84, out_cols_=8400
    int             out_rows_{300};
    int             out_cols_{6};
    // True when out_cols_ >> out_rows_ (transposed YOLOv8 format requires CPU NMS)
    bool            is_yolov8_format_{false};

    // Detection thresholds
    float           confidence_threshold_{0.50f};
    float           nms_threshold_{0.45f};

    // Class filter (empty = keep all)
    std::vector<int64_t>        allowed_classes_raw_{};
    std::unordered_set<int>     allowed_classes_{};
    std::vector<std::string>    class_names_{};

    // Topic / QoS
    std::string     sub_topic_{"/arducam/left/nitros_image"};
    std::string     sub_reliability_{"best_effort"};
    std::string     sub_durability_{"volatile"};
    int             sub_depth_{10};
    std::string     pub_topic_{"/detections"};
    std::string     pub_reliability_{"best_effort"};
    std::string     pub_durability_{"volatile"};
    int             pub_depth_{10};



    // --- TensorRT resources (all GPU resident) ----------------------------------------------

    TRTLogger                                       trt_logger_;
    std::unique_ptr<nvinfer1::IRuntime>             trt_runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>          trt_engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>    trt_context_;
    cudaStream_t                                    cuda_stream_{nullptr};

    // --- Device-side I/O buffers - never touch host memory in the hot path! ---
    void* d_input_{nullptr}; // [1, 3, H, W] float16/float32    ---  Device buffer for input image
    void* d_output_{nullptr}; // output buffer — sized to out_rows_ * out_cols_ * sizeof(float)

    // --- VIC / NvBufSurface workspace -------------------------------------------------------
    NvBufSurface*   vic_dst_surf_{nullptr}; // VIC output surface (resized/formatted frames for TensorRT input)

    // ---ROS2 I/O -----------------------------------------------------------------------------

    // ManagedNitrosSubscriber receives GPU tensors without any CPU memcpy. 
    // The upstream node (eg. isaac_ros_h264_decoder + DNN encoder) feeds this.
    std::shared_ptr<
        nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
            nvidia::isaac_ros::nitros::NitrosTensorListView>>   nitros_sub_;

    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr   detection_pub_;

    // --- Resource monitoring (1Hz wall timer) ---
    rclcpp::TimerBase::SharedPtr resource_monitor_timer_;


    // --- Pipeline Methods -------------------------------------------------------------------------
    
    // Main callback -- receive GPU-resident tensor list from Nitros transport
    void InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView& msg);
    
    // Pre-processing: YUV -> RGB + resize via VIC (hardware engine, zero-copy GPU)
    // fallback to CUDA kernel when use_vic_preprocessing_ is false or on failure
    void PreprocessVIC(NvBufSurface* src_surf, void* d_dst_float);
    void PreprocessCUDA(const void* d_src_nv12, int src_w, int src_h, void* d_dst_float);

    // Inference: TRT enqueueV3 - fully async, returns immediately after queuing work on GPU
    void RunInference();

    // Post-processing: decode model output, filter by confidence, apply NMS if needed.
    // Branches automatically based on is_yolov8_format_.
    std::vector<Object> DecodeOutputGPU(const void* d_output);

   


    // Publishers for detections and visualization markers
    void PublishDetections(
        const std::vector<Object> & objects,
        const std_msgs::msg::Header & header);
    // TRT setup
    void BuildEngine();
    // VIC workspace allocation
    void AllocVICSurface();
    // Load class names from file
    void LoadClassNames();
    // Build a rclcpp::QoS from string settings
    rclcpp::QoS BuildQoS(const std::string & reliability,
                         const std::string & durability,
                         int depth) const;
    
};


} // namespace yolo26
}  // namespace isaac_ros
}  // namespace nvidia


