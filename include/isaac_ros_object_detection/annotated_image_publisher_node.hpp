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
 * @file: annotation_image_publisher_node.hpp
 * @brief: Header for the AnnotatedImagePublisherNode composable ROS2 node.
 * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
 * @company: Sintez.LLC
 * @date: 2026-03-01
 * 
 * ROS2 composable node that subscribes to an image topic (raw, compressed, or
 * NITROS) plus a Detection2DArray, overlays bounding-box rectangles and text
 * labels entirely on the GPU, and publishes the annotated frame together with
 * a re-stamped CameraInfo so RViz2 can display it out-of-the-box.
 * 
 * Hardware acceleration:
 *   • VIC  — NITROS→RGBA colour-format conversion (NvBufSurfTransform)
 *   • CUDA — bbox drawing, glyph-atlas text stamping, pixel-format helpers,
 *           nvJPEG GPU JPEG decode/encode
 *   • CPU  — one-time atlas build (cv::putText at startup), optional OpenCV
 *             JPEG fallback when nvJPEG is absent
 * ─────────────────────────────────────────────────────────────────────────────
*/

#ifndef ISAAC_ROS_ANNOTATED_IMAGE_PUBLISHER_NODE_HPP_
#define ISAAC_ROS_ANNOTATED_IMAGE_PUBLISHER_NODE_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

// ROS2
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

// Isaac ROS NITROS zero-copy image transport
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"

// CUDA / Jetson HW
#include <cuda_runtime_api.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#ifdef WITH_NVJPEG
#include <nvjpeg.h>
#endif

// Annotation GPU kernel API
#include "isaac_ros_object_detection/annotation_kernels.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace object_detection
{

// ─────────────────────────────────────────────────────────────────────────────
// Short-hand message types
// ─────────────────────────────────────────────────────────────────────────────
using ImageMsg         = sensor_msgs::msg::Image;
using CompressedMsg    = sensor_msgs::msg::CompressedImage;
using CameraInfoMsg    = sensor_msgs::msg::CameraInfo;
using Detection2DArray = vision_msgs::msg::Detection2DArray;

// ─────────────────────────────────────────────────────────────────────────────
// AnnotatedImagePublisherNode
// ─────────────────────────────────────────────────────────────────────────────

class AnnotatedImagePublisherNode : public rclcpp::Node
{
public:
    explicit AnnotatedImagePublisherNode(
        const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

    ~AnnotatedImagePublisherNode();

private:
    // ── Enums ─────────────────────────────────────────────────────────────────

    enum class ImageTransport { RAW, COMPRESSED, NITROS };
    enum class PubTransport   { RAW, COMPRESSED };

    // ── Parameters ────────────────────────────────────────────────────────────

    ImageTransport img_transport_{ImageTransport::COMPRESSED};
    PubTransport   pub_transport_{PubTransport::COMPRESSED};

    std::string det_topic_name_         {"/object_detection/detections"};
    std::string det_reliability_        {"best_effort"};
    std::string det_durability_         {"volatile"};
    int         det_depth_              {10};

    std::string img_topic_name_         {"/arducam/left/image/compressed"};
    std::string img_transport_str_      {"compressed"};
    std::string img_reliability_        {"reliable"};
    std::string img_durability_         {"volatile"};
    int         img_depth_              {10};

    std::string cam_info_topic_         {"/arducam/left/image/camera_info"};
    std::string cam_reliability_        {"best_effort"};
    std::string cam_durability_         {"volatile"};
    int         cam_depth_              {10};

    std::string pub_topic_name_prefix_  {"/object_detection/annotated_image"};
    std::string pub_transport_str_      {"compressed"};
    std::string pub_reliability_        {"best_effort"};
    std::string pub_durability_         {"volatile"};
    int         pub_depth_              {10};

    int   model_input_width_            {640};
    int   model_input_height_           {640};
    int   num_classes_                  {80};

    bool    draw_labels_                {true};
    bool    draw_confidence_            {true};
    float   confidence_threshold_       {0.5f};
    double  font_size_                  {0.5};
    int     font_thickness_             {1};
    int     box_thickness_              {2};
    // box_color: [-1,-1,-1] → per-class; otherwise BGR fixed colour
    int     box_color_b_                {0};
    int     box_color_g_                {255};
    int     box_color_r_                {0};

    // ── ROS2 subscribers ──────────────────────────────────────────────────────

    rclcpp::Subscription<Detection2DArray>::SharedPtr         det_sub_;
    rclcpp::Subscription<ImageMsg>::SharedPtr                 raw_sub_;
    rclcpp::Subscription<CompressedMsg>::SharedPtr            comp_sub_;
    std::shared_ptr<
        nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
            nvidia::isaac_ros::nitros::NitrosImageView>>      nitros_sub_;
    rclcpp::Subscription<CameraInfoMsg>::SharedPtr            cam_info_sub_;

    // ── ROS2 publishers ───────────────────────────────────────────────────────

    rclcpp::Publisher<ImageMsg>::SharedPtr         raw_pub_;
    rclcpp::Publisher<CompressedMsg>::SharedPtr    comp_pub_;
    rclcpp::Publisher<CameraInfoMsg>::SharedPtr    cam_info_pub_;

    // ── Detection cache (latest, thread-safe) ─────────────────────────────────

    Detection2DArray::SharedPtr last_det_;
    std::mutex                  det_mtx_;

    // ── CameraInfo cache ──────────────────────────────────────────────────────

    CameraInfoMsg::SharedPtr    cached_cam_info_;
    std::mutex                  cam_info_mtx_;

    // ── CUDA resources ────────────────────────────────────────────────────────

    cudaStream_t   stream_       {nullptr};

    // Device BGR buffer (lazy grow-on-first-frame, never shrinks)
    uint8_t*       d_bgr_        {nullptr};
    size_t         d_bgr_size_   {0};

    // Pinned host staging buffer (DtoH/HtoD without pageable penalty)
    uint8_t*       h_staging_    {nullptr};
    size_t         h_staging_sz_ {0};

    // ── VIC resources (NITROS path) ───────────────────────────────────────────

    NvBufSurface*  vic_rgba_surf_  {nullptr};   // VIC output: RGBA, same w×h as source
    int            vic_surf_w_     {0};
    int            vic_surf_h_     {0};
    bool           use_vic_        {false};      // set on first NITROS frame
    bool           vic_probed_     {false};      // probe done?

    // ── nvJPEG handles ─────────────────────────────────────────────────────────

#ifdef WITH_NVJPEG
    nvjpegHandle_t          nvjpeg_handle_   {nullptr};
    nvjpegJpegState_t       nvjpeg_state_    {nullptr};
    nvjpegEncoderState_t    enc_state_       {nullptr};
    nvjpegEncoderParams_t   enc_params_      {nullptr};

    // GPU plane buffers for nvJPEG decode output (BGR planar)
    uint8_t*                d_nvjpeg_b_      {nullptr};
    uint8_t*                d_nvjpeg_g_      {nullptr};
    uint8_t*                d_nvjpeg_r_      {nullptr};
    size_t                  d_nvjpeg_plane_  {0};   // bytes per plane
#endif

    // ── Glyph atlas ───────────────────────────────────────────────────────────

    cudaTextureObject_t     atlas_tex_       {0};
    cudaArray_t             atlas_arr_       {nullptr};
    int                     glyph_w_         {0};
    int                     glyph_h_         {0};

    // Per-frame device buffers for text drawing
    uint8_t*   d_label_chars_  {nullptr};   // packed chars, all detections
    int*       d_label_offsets_{nullptr};   // [MAX_DETS]
    int*       d_label_lens_   {nullptr};   // [MAX_DETS]
    int*       d_label_x_      {nullptr};   // [MAX_DETS]
    int*       d_label_y_      {nullptr};   // [MAX_DETS]
    size_t     d_label_chars_sz_{0};
    int        d_label_det_cap_ {0};        // capacity of per-det device arrays (in # detections)

    // ── Lifecycle helpers ──────────────────────────────────────────────────────

    void InitGlyphAtlas();
    void InitNvJpeg();
    void AllocVICSurface(int w, int h);
    void EnsureDeviceBuf(size_t bytes);
    void EnsureStagingBuf(size_t bytes);
    void EnsureNvJpegPlaneBufs(size_t plane_bytes);
    void EnsureLabelDevBufs(int max_dets, int max_chars);

    // ── Callbacks ──────────────────────────────────────────────────────────────

    void DetectionCallback(Detection2DArray::SharedPtr msg);
    void CamInfoCallback(CameraInfoMsg::SharedPtr msg);

    void RawImageCallback(ImageMsg::SharedPtr msg);
    void CompressedImageCallback(CompressedMsg::SharedPtr msg);
    void NitrosImageCallback(
        const nvidia::isaac_ros::nitros::NitrosImageView & view);

    // ── Core annotation & publish ─────────────────────────────────────────────

    // Called by all input paths once d_bgr_ holds the current frame (packed BGR).
    void AnnotateAndPublish(
        int               img_w,
        int               img_h,
        const std_msgs::msg::Header & header);

};

}  // namespace object_detection
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_ANNOTATED_IMAGE_PUBLISHER_NODE_HPP_


 