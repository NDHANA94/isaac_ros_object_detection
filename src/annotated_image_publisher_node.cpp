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
 * @file: annotated_image_publisher_node.cpp
 * @brief: Implementation of the AnnotatedImagePublisherNode composable ROS2 node.
 * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
 * @company: Sintez.LLC
 * @date: 2026-03-01
 * 
 * This file implements the AnnotatedImagePublisherNode, which subscribes to an 
 * image topic, a Detection2DArray topic, and a CameraInfo topic, and publishes 
 * annotated images with bounding boxes and labels overlaid.
 * 
 * The node uses hardware acceleration with VIC for color-format conversion and 
 * CUDA for drawing annotations, and is designed to be flexible and configurable 
 * via ROS2 parameters.
 * 
 * The implementation includes the constructor, destructor, and the main callback 
 * for processing incoming images and detections.  The node is designed to be 
 * thread-safe and efficient for real-time applications.
 * 
 * Pipeline block diagram: TODO
 * 
 * 
 * 
 * 
 * ─────────────────────────────────────────────────────────────────────────────
*/


#include "isaac_ros_object_detection/annotated_image_publisher_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "rclcpp_components/register_node_macro.hpp"
#include "std_msgs/msg/header.hpp"

namespace nvidia { namespace isaac_ros { namespace object_detection {

// ─────────────────────────────────────────────────────────────────────────────
// Utilities (file-scope)
// ─────────────────────────────────────────────────────────────────────────────

static rclcpp::QoS build_qos(const std::string & rel,
                              const std::string & dur, int depth)
{
    rclcpp::QoS q{static_cast<size_t>(depth)};
    (rel == "reliable") ? q.reliable()   : q.best_effort();
    (dur == "transient_local") ? q.transient_local() : q.durability_volatile();
    return q;
}

// HSV hue-cycle → BGR per class
static void class_color(int cls_id, int num_classes,
                         uint8_t & b, uint8_t & g, uint8_t & r)
{
    const float hue = (360.f * cls_id / std::max(1, num_classes));
    // simple HSV(hue,1,1) → RGB
    const float h6 = hue / 60.f;
    const int   hi = static_cast<int>(h6) % 6;
    const float f  = h6 - std::floor(h6);
    const float q_ = 1.f - f;
    float rv = 0, gv = 0, bv = 0;
    switch (hi) {
        case 0: rv=1; gv=f;  bv=0; break;
        case 1: rv=q_; gv=1; bv=0; break;
        case 2: rv=0; gv=1;  bv=f; break;
        case 3: rv=0; gv=q_; bv=1; break;
        case 4: rv=f; gv=0;  bv=1; break;
        case 5: rv=1; gv=0;  bv=q_; break;
    }
    r = static_cast<uint8_t>(rv * 255);
    g = static_cast<uint8_t>(gv * 255);
    b = static_cast<uint8_t>(bv * 255);
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
AnnotatedImagePublisherNode::AnnotatedImagePublisherNode(
    const rclcpp::NodeOptions & options)
: Node("annotated_image_publisher_node", options)
{
    // ── Declare & read parameters ─────────────────────────────────────────
    img_transport_str_ = declare_parameter<std::string>(
        "sub_topic_image.transport", "compressed");
    img_topic_name_    = declare_parameter<std::string>(
        "sub_topic_image.topic_name", "/arducam/left/image/compressed");
    img_reliability_   = declare_parameter<std::string>(
        "sub_topic_image.sub_reliability", "reliable");
    img_durability_    = declare_parameter<std::string>(
        "sub_topic_image.sub_durability", "volatile");
    img_depth_         = declare_parameter<int>(
        "sub_topic_image.sub_depth", 10);

    det_topic_name_  = declare_parameter<std::string>(
        "sub_topic_detections.topic_name", "/object_detection/detections");
    det_reliability_ = declare_parameter<std::string>(
        "sub_topic_detections.sub_reliability", "best_effort");
    det_durability_  = declare_parameter<std::string>(
        "sub_topic_detections.sub_durability", "volatile");
    det_depth_       = declare_parameter<int>(
        "sub_topic_detections.sub_depth", 10);

    cam_info_topic_  = declare_parameter<std::string>(
        "sub_topic_camera_info.topic_name", "/arducam/left/image/camera_info");
    cam_reliability_ = declare_parameter<std::string>(
        "sub_topic_camera_info.sub_reliability", "best_effort");
    cam_durability_  = declare_parameter<std::string>(
        "sub_topic_camera_info.sub_durability", "volatile");
    cam_depth_       = declare_parameter<int>(
        "sub_topic_camera_info.sub_depth", 10);

    pub_topic_name_prefix_    = declare_parameter<std::string>(
        "pub_topic_image.topic_name_prefix", "/object_detection/annotated_image");
    pub_transport_str_ = declare_parameter<std::string>(
        "pub_topic_image.transport", "compressed");
    pub_reliability_   = declare_parameter<std::string>(
        "pub_topic_image.pub_reliability", "best_effort");
    pub_durability_    = declare_parameter<std::string>(
        "pub_topic_image.pub_durability", "volatile");
    pub_depth_         = declare_parameter<int>(
        "pub_topic_image.pub_depth", 10);

    // detection_model_input_width / detection_model_input_height are no longer required:
    // bounding boxes from ObjectDetectionNode are published normalised to [0, 1]
    // relative to the model input size, so scaling by the actual image dimensions
    // (already known from the incoming image message) is sufficient.
    num_classes_        = declare_parameter<int>("num_classes", 80);

    draw_labels_         = declare_parameter<bool>("draw_labels",    true);
    draw_confidence_     = declare_parameter<bool>("draw_confidence", true);
    confidence_threshold_= static_cast<float>(
        declare_parameter<double>("confidence_threshold", 0.5));
    font_size_           = declare_parameter<double>("font_size",      0.5);
    font_thickness_      = declare_parameter<int>("font_thickness",    1);
    box_thickness_       = declare_parameter<int>("box_thickness",     2);
    auto box_color = declare_parameter<std::vector<long>>("box_color",
        std::vector<long>{0, 255, 0});
    box_color_b_ = static_cast<int>(box_color[0]);
    box_color_g_ = static_cast<int>(box_color[1]);
    box_color_r_ = static_cast<int>(box_color[2]);

    // Resolve transport enum
    if      (img_transport_str_ == "raw")        img_transport_ = ImageTransport::RAW;
    else if (img_transport_str_ == "compressed") img_transport_ = ImageTransport::COMPRESSED;
    else if (img_transport_str_ == "nitros")     img_transport_ = ImageTransport::NITROS;
    else {
        RCLCPP_WARN(get_logger(),
            "Unknown sub_topic_image.transport '%s', defaulting to 'compressed'",
            img_transport_str_.c_str());
        img_transport_ = ImageTransport::COMPRESSED;
    }

    pub_transport_ = (pub_transport_str_ == "raw") ? PubTransport::RAW : PubTransport::COMPRESSED;

    // ── CUDA stream ───────────────────────────────────────────────────────
    cudaError_t ce = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (ce != cudaSuccess)
        throw std::runtime_error(std::string("cudaStreamCreate: ") + cudaGetErrorString(ce));

    // ── Initialize subsystems ─────────────────────────────────────────────
    InitGlyphAtlas();
    InitNvJpeg();

    // ── Publishers ────────────────────────────────────────────────────────
    const std::string cam_info_pub_topic = pub_topic_name_prefix_ + "/camera_info";
    cam_info_pub_ = create_publisher<CameraInfoMsg>(
        cam_info_pub_topic,
        build_qos(pub_reliability_, pub_durability_, pub_depth_));

    if (pub_transport_ == PubTransport::COMPRESSED) {
        comp_pub_ = create_publisher<CompressedMsg>(
            pub_topic_name_prefix_ + "/compressed",
            build_qos(pub_reliability_, pub_durability_, pub_depth_));
    } else {
        raw_pub_ = create_publisher<ImageMsg>(
            pub_topic_name_prefix_ + "/raw",
            build_qos(pub_reliability_, pub_durability_, pub_depth_));
    }

    // ── Detection subscriber ──────────────────────────────────────────────
    det_sub_ = create_subscription<Detection2DArray>(
        det_topic_name_,
        build_qos(det_reliability_, det_durability_, det_depth_),
        [this](Detection2DArray::SharedPtr msg){ DetectionCallback(msg); });

    // ── CameraInfo subscriber ─────────────────────────────────────────────
    cam_info_sub_ = create_subscription<CameraInfoMsg>(
        cam_info_topic_,
        build_qos(cam_reliability_, cam_durability_, cam_depth_),
        [this](CameraInfoMsg::SharedPtr msg){ CamInfoCallback(msg); });

    // ── Image subscriber (transport-dependent) ────────────────────────────
    if (img_transport_ == ImageTransport::RAW) {
        raw_sub_ = create_subscription<ImageMsg>(
            img_topic_name_,
            build_qos(img_reliability_, img_durability_, img_depth_),
            [this](ImageMsg::SharedPtr msg){ RawImageCallback(msg); });
    } else if (img_transport_ == ImageTransport::COMPRESSED) {
        comp_sub_ = create_subscription<CompressedMsg>(
            img_topic_name_,
            build_qos(img_reliability_, img_durability_, img_depth_),
            [this](CompressedMsg::SharedPtr msg){ CompressedImageCallback(msg); });
    } else {  // NITROS
        nitros_sub_ = std::make_shared<
            nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
                nvidia::isaac_ros::nitros::NitrosImageView>>(
            this,
            img_topic_name_,
            nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name,
            [this](const nvidia::isaac_ros::nitros::NitrosImageView & view){
                NitrosImageCallback(view); },
            nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{},
            build_qos(img_reliability_, img_durability_, img_depth_));
    }

    RCLCPP_INFO(get_logger(),
        "AnnotatedImagePublisherNode ready | transport=%s | pub=%s | det=%s",
        img_transport_str_.c_str(), pub_topic_name_prefix_.c_str(), det_topic_name_.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────────────────
AnnotatedImagePublisherNode::~AnnotatedImagePublisherNode()
{
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }

    if (d_bgr_)         { cudaFree(d_bgr_);         d_bgr_        = nullptr; }
    if (h_staging_)     { cudaFreeHost(h_staging_);  h_staging_    = nullptr; }

    if (d_label_chars_)  { cudaFree(d_label_chars_);  d_label_chars_  = nullptr; }
    if (d_label_offsets_){ cudaFree(d_label_offsets_); d_label_offsets_= nullptr; }
    if (d_label_lens_)   { cudaFree(d_label_lens_);   d_label_lens_   = nullptr; }
    if (d_label_x_)      { cudaFree(d_label_x_);      d_label_x_      = nullptr; }
    if (d_label_y_)      { cudaFree(d_label_y_);      d_label_y_      = nullptr; }

    if (atlas_tex_) { cudaDestroyTextureObject(atlas_tex_); atlas_tex_ = 0; }
    if (atlas_arr_) { cudaFreeArray(atlas_arr_);            atlas_arr_ = nullptr; }

    if (vic_rgba_surf_) { NvBufSurfaceDestroy(vic_rgba_surf_); vic_rgba_surf_ = nullptr; }

#ifdef WITH_NVJPEG
    if (enc_params_)    { nvjpegEncoderParamsDestroy(enc_params_);   enc_params_   = nullptr; }
    if (enc_state_)     { nvjpegEncoderStateDestroy(enc_state_);     enc_state_    = nullptr; }
    if (nvjpeg_state_)  { nvjpegJpegStateDestroy(nvjpeg_state_);     nvjpeg_state_ = nullptr; }
    if (nvjpeg_handle_) { nvjpegDestroy(nvjpeg_handle_);             nvjpeg_handle_= nullptr; }
    if (d_nvjpeg_b_)    { cudaFree(d_nvjpeg_b_); d_nvjpeg_b_ = nullptr; }
    if (d_nvjpeg_g_)    { cudaFree(d_nvjpeg_g_); d_nvjpeg_g_ = nullptr; }
    if (d_nvjpeg_r_)    { cudaFree(d_nvjpeg_r_); d_nvjpeg_r_ = nullptr; }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// InitGlyphAtlas — build a CPU glyph atlas via cv::putText, upload to GPU
// ─────────────────────────────────────────────────────────────────────────────
void AnnotatedImagePublisherNode::InitGlyphAtlas()
{
    const int  cv_font   = cv::FONT_HERSHEY_SIMPLEX;
    const int  num_chars = 95;  // ASCII 32..126
    int        baseline  = 0;
    cv::Size   sz = cv::getTextSize("A", cv_font, font_size_, font_thickness_, &baseline);
    glyph_w_ = sz.width  + 2;   // +2px padding
    glyph_h_ = sz.height + baseline + 2;

    cv::Mat atlas(glyph_h_, glyph_w_ * num_chars, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < num_chars; ++i) {
        char buf[2] = { static_cast<char>(32 + i), '\0' };
        int dummy = 0;
        cv::Size gs = cv::getTextSize(buf, cv_font, font_size_, font_thickness_, &dummy);
        cv::putText(atlas, buf,
            cv::Point(i * glyph_w_ + 1, glyph_h_ - baseline - 1),
            cv_font, font_size_, cv::Scalar(255), font_thickness_);
    }

    // Upload to CUDA array + create texture
    cudaChannelFormatDesc cfd = cudaCreateChannelDesc<uint8_t>();
    cudaError_t ce = cudaMallocArray(&atlas_arr_,
        &cfd, atlas.cols, atlas.rows);
    if (ce != cudaSuccess) {
        RCLCPP_ERROR(get_logger(), "cudaMallocArray atlas: %s", cudaGetErrorString(ce));
        return;
    }
    cudaMemcpy2DToArray(atlas_arr_, 0, 0,
        atlas.data, atlas.step,
        atlas.cols * sizeof(uint8_t), atlas.rows,
        cudaMemcpyHostToDevice);

    cudaResourceDesc  rd{};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = atlas_arr_;
    cudaTextureDesc   td{};
    td.addressMode[0]  = cudaAddressModeClamp;
    td.addressMode[1]  = cudaAddressModeClamp;
    td.filterMode      = cudaFilterModePoint;
    td.readMode        = cudaReadModeElementType;
    td.normalizedCoords= 0;
    cudaCreateTextureObject(&atlas_tex_, &rd, &td, nullptr);
    RCLCPP_INFO(get_logger(),
        "Glyph atlas ready: %dx%d  glyph=%dx%d",
        atlas.cols, atlas.rows, glyph_w_, glyph_h_);
}

// ─────────────────────────────────────────────────────────────────────────────
// InitNvJpeg
// ─────────────────────────────────────────────────────────────────────────────
void AnnotatedImagePublisherNode::InitNvJpeg()
{
#ifdef WITH_NVJPEG
    if (nvjpegCreate(NVJPEG_BACKEND_DEFAULT, nullptr, &nvjpeg_handle_) != NVJPEG_STATUS_SUCCESS) {
        RCLCPP_WARN(get_logger(), "nvjpegCreate failed — GPU JPEG disabled");
        nvjpeg_handle_ = nullptr; return;
    }
    nvjpegJpegStateCreate(nvjpeg_handle_, &nvjpeg_state_);
    nvjpegEncoderStateCreate(nvjpeg_handle_, &enc_state_, stream_);
    nvjpegEncoderParamsCreate(nvjpeg_handle_, &enc_params_, stream_);
    nvjpegEncoderParamsSetQuality(enc_params_, 85, stream_);
    nvjpegEncoderParamsSetSamplingFactors(enc_params_, NVJPEG_CSS_420, stream_);
    RCLCPP_INFO(get_logger(), "nvJPEG initialized (GPU JPEG encode/decode enabled)");
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// AllocVICSurface — allocate VIC RGBA destination surface
// ─────────────────────────────────────────────────────────────────────────────
void AnnotatedImagePublisherNode::AllocVICSurface(int w, int h)
{
    if (vic_rgba_surf_ && vic_surf_w_ == w && vic_surf_h_ == h) return;
    if (vic_rgba_surf_) { NvBufSurfaceDestroy(vic_rgba_surf_); vic_rgba_surf_ = nullptr; }

    NvBufSurfaceCreateParams cp{};
    cp.gpuId       = 0;
    cp.width       = w;
    cp.height      = h;
    cp.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    cp.layout      = NVBUF_LAYOUT_PITCH;
    cp.memType     = NVBUF_MEM_CUDA_DEVICE;
    if (NvBufSurfaceCreate(&vic_rgba_surf_, 1, &cp) != 0) {
        RCLCPP_ERROR(get_logger(), "AllocVICSurface failed for %dx%d", w, h);
        vic_rgba_surf_ = nullptr; return;
    }
    vic_surf_w_ = w; vic_surf_h_ = h;
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffer helpers
// ─────────────────────────────────────────────────────────────────────────────
void AnnotatedImagePublisherNode::EnsureDeviceBuf(size_t bytes)
{
    if (bytes <= d_bgr_size_) return;
    if (d_bgr_) cudaFree(d_bgr_);
    cudaMalloc(&d_bgr_, bytes);
    d_bgr_size_ = bytes;
}

void AnnotatedImagePublisherNode::EnsureStagingBuf(size_t bytes)
{
    if (bytes <= h_staging_sz_) return;
    if (h_staging_) cudaFreeHost(h_staging_);
    cudaHostAlloc(&h_staging_, bytes, cudaHostAllocDefault);
    h_staging_sz_ = bytes;
}

void AnnotatedImagePublisherNode::EnsureNvJpegPlaneBufs(size_t plane_bytes)
{
#ifdef WITH_NVJPEG
    if (plane_bytes <= d_nvjpeg_plane_) return;
    if (d_nvjpeg_b_) cudaFree(d_nvjpeg_b_);
    if (d_nvjpeg_g_) cudaFree(d_nvjpeg_g_);
    if (d_nvjpeg_r_) cudaFree(d_nvjpeg_r_);
    cudaMalloc(&d_nvjpeg_b_, plane_bytes);
    cudaMalloc(&d_nvjpeg_g_, plane_bytes);
    cudaMalloc(&d_nvjpeg_r_, plane_bytes);
    d_nvjpeg_plane_ = plane_bytes;
#else
    (void)plane_bytes;
#endif
}

void AnnotatedImagePublisherNode::EnsureLabelDevBufs(int max_dets, int max_chars)
{
    // Grow per-detection arrays if needed
    if (max_dets > d_label_det_cap_) {
        if (d_label_offsets_) cudaFree(d_label_offsets_);
        if (d_label_lens_)    cudaFree(d_label_lens_);
        if (d_label_x_)       cudaFree(d_label_x_);
        if (d_label_y_)       cudaFree(d_label_y_);
        cudaMalloc(&d_label_offsets_, max_dets * sizeof(int));
        cudaMalloc(&d_label_lens_,    max_dets * sizeof(int));
        cudaMalloc(&d_label_x_,       max_dets * sizeof(int));
        cudaMalloc(&d_label_y_,       max_dets * sizeof(int));
        d_label_det_cap_ = max_dets;
    }
    if (static_cast<int>(max_chars) > static_cast<int>(d_label_chars_sz_)) {
        if (d_label_chars_) cudaFree(d_label_chars_);
        cudaMalloc(&d_label_chars_, max_chars);
        d_label_chars_sz_ = max_chars;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detection & CameraInfo callbacks (just cache the latest)
// ─────────────────────────────────────────────────────────────────────────────
void AnnotatedImagePublisherNode::DetectionCallback(Detection2DArray::SharedPtr msg)
{
    std::lock_guard<std::mutex> lk(det_mtx_);
    last_det_ = std::move(msg);
}

void AnnotatedImagePublisherNode::CamInfoCallback(CameraInfoMsg::SharedPtr msg)
{
    std::lock_guard<std::mutex> lk(cam_info_mtx_);
    cached_cam_info_ = std::move(msg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Raw image callback
// ─────────────────────────────────────────────────────────────────────────────
void AnnotatedImagePublisherNode::RawImageCallback(ImageMsg::SharedPtr msg)
{
    const int w = static_cast<int>(msg->width);
    const int h = static_cast<int>(msg->height);
    const size_t frame_bytes = static_cast<size_t>(w) * h * 3;

    EnsureDeviceBuf(frame_bytes);
    EnsureStagingBuf(frame_bytes);

    // HtoD via pinned staging
    if (msg->encoding == "bgr8" || msg->encoding == "rgb8") {
        std::memcpy(h_staging_, msg->data.data(), frame_bytes);
        cudaMemcpyAsync(d_bgr_, h_staging_, frame_bytes,
                        cudaMemcpyHostToDevice, stream_);
        if (msg->encoding == "rgb8")
            cuda_rgb_to_bgr_inplace(d_bgr_, w, h, stream_);
    } else if (msg->encoding == "mono8") {
        // Replicate grey channel to BGR
        const uint8_t * src = msg->data.data();
        uint8_t * dst = h_staging_;
        for (int i = 0; i < w * h; ++i) {
            dst[i*3+0] = dst[i*3+1] = dst[i*3+2] = src[i];
        }
        cudaMemcpyAsync(d_bgr_, h_staging_, frame_bytes,
                        cudaMemcpyHostToDevice, stream_);
    } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "Unsupported raw encoding '%s'", msg->encoding.c_str());
        return;
    }
    AnnotateAndPublish(w, h, msg->header);
}

// ─────────────────────────────────────────────────────────────────────────────
// Compressed image callback
// ─────────────────────────────────────────────────────────────────────────────
void AnnotatedImagePublisherNode::CompressedImageCallback(CompressedMsg::SharedPtr msg)
{
#ifdef WITH_NVJPEG
    if (nvjpeg_handle_) {
        // GPU decode path
        nvjpegImage_t nv_img{};
        // We'll decode into planar BGR (nvJPEG channel order: B=0, G=1, R=2)
        // First pass: get image info
        int nComponents = 0;
        nvjpegChromaSubsampling_t subsampling{};
        int widths[NVJPEG_MAX_COMPONENT]  = {};
        int heights[NVJPEG_MAX_COMPONENT] = {};
        if (nvjpegGetImageInfo(nvjpeg_handle_,
                msg->data.data(), msg->data.size(),
                &nComponents, &subsampling, widths, heights)
            != NVJPEG_STATUS_SUCCESS)
        {
            goto cpu_decode;
        }
        const int w = widths[0], h = heights[0];
        const size_t plane_bytes = static_cast<size_t>(w) * h;
        EnsureNvJpegPlaneBufs(plane_bytes);
        EnsureDeviceBuf(plane_bytes * 3);

        nv_img.channel[0] = d_nvjpeg_b_;
        nv_img.channel[1] = d_nvjpeg_g_;
        nv_img.channel[2] = d_nvjpeg_r_;
        nv_img.pitch[0]   = w;
        nv_img.pitch[1]   = w;
        nv_img.pitch[2]   = w;

        if (nvjpegDecode(nvjpeg_handle_, nvjpeg_state_,
                msg->data.data(), msg->data.size(),
                NVJPEG_OUTPUT_BGR, &nv_img, stream_)
            != NVJPEG_STATUS_SUCCESS)
        {
            goto cpu_decode;
        }
        // Interleave planar BGR → packed BGR
        cuda_planar_to_packed_bgr(
            d_nvjpeg_b_, d_nvjpeg_g_, d_nvjpeg_r_,
            d_bgr_, w, h, stream_);

        AnnotateAndPublish(w, h, msg->header);
        return;

        cpu_decode:;
    }
#endif
    // CPU fallback: cv::imdecode → HtoD
    std::vector<uint8_t> buf(msg->data.begin(), msg->data.end());
    cv::Mat decoded = cv::imdecode(buf, cv::IMREAD_COLOR);  // BGR
    if (decoded.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cv::imdecode failed");
        return;
    }
    const int w = decoded.cols, h = decoded.rows;
    const size_t frame_bytes = static_cast<size_t>(w) * h * 3;
    EnsureDeviceBuf(frame_bytes);
    EnsureStagingBuf(frame_bytes);

    std::memcpy(h_staging_, decoded.data, frame_bytes);
    cudaMemcpyAsync(d_bgr_, h_staging_, frame_bytes,
                    cudaMemcpyHostToDevice, stream_);
    AnnotateAndPublish(w, h, msg->header);
}

// ─────────────────────────────────────────────────────────────────────────────
// NITROS image callback
// ─────────────────────────────────────────────────────────────────────────────
void AnnotatedImagePublisherNode::NitrosImageCallback(
    const nvidia::isaac_ros::nitros::NitrosImageView & view)
{
    const int w = static_cast<int>(view.GetWidth());
    const int h = static_cast<int>(view.GetHeight());
    const size_t frame_bytes = static_cast<size_t>(w) * h * 3;

    EnsureDeviceBuf(frame_bytes);

    // Build ROS header
    std_msgs::msg::Header header;
    header.stamp.sec     = view.GetTimestampSeconds();
    header.stamp.nanosec = view.GetTimestampNanoseconds();
    header.frame_id      = view.GetFrameId();

    const uint8_t * d_src = view.GetGpuData();

    // Try VIC on first frame; fall back to CUDA kernel
    if (!vic_probed_) {
        vic_probed_ = true;
        AllocVICSurface(w, h);
        if (vic_rgba_surf_) {
            NvBufSurface * src_surf = nullptr;
            // The NITROS RGB8 buffer from ArducamB0573 is a plain cudaMalloc
            // buffer via the TypeAdapter — NOT an NvBufSurface.  Attempting
            // NvBufSurfTransform on it will fail.  We detect this by trying
            // a dummy transform and checking the return code.
            NvBufSurfaceCreateParams cp{};
            cp.gpuId = 0; cp.width = 1; cp.height = 1;
            cp.colorFormat = NVBUF_COLOR_FORMAT_RGB;
            cp.layout = NVBUF_LAYOUT_PITCH;
            cp.memType = NVBUF_MEM_CUDA_DEVICE;
            if (NvBufSurfaceCreate(&src_surf, 1, &cp) == 0) {
                NvBufSurfTransformParams tp{};
                tp.transform_flag   = NVBUFSURF_TRANSFORM_FILTER;
                tp.transform_filter = NvBufSurfTransformInter_Bilinear;
                use_vic_ = (NvBufSurfTransform(src_surf, vic_rgba_surf_, &tp)
                            == NvBufSurfTransformError_Success);
                NvBufSurfaceDestroy(src_surf);
            }
            if (!use_vic_) {
                RCLCPP_WARN(get_logger(),
                    "VIC NvBufSurfTransform probe failed — using CUDA RGB→BGR fallback");
            } else {
                RCLCPP_INFO(get_logger(), "VIC probe succeeded");
            }
        }
    }

    if (use_vic_ && vic_rgba_surf_) {
        NvBufSurface * src_surf =
            const_cast<NvBufSurface *>(reinterpret_cast<const NvBufSurface *>(d_src));
        NvBufSurfTransformParams tp{};
        tp.transform_flag   = NVBUFSURF_TRANSFORM_FILTER;
        tp.transform_filter = NvBufSurfTransformInter_Bilinear;
        if (NvBufSurfTransform(src_surf, vic_rgba_surf_, &tp)
            != NvBufSurfTransformError_Success)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "VIC transform failed this frame — using CUDA fallback");
            goto nitros_cuda_fallback;
        }
        // RGBA → packed BGR
        EnsureDeviceBuf(frame_bytes);
        cuda_rgba_to_bgr_packed(
            static_cast<const uint8_t *>(vic_rgba_surf_->surfaceList[0].dataPtr),
            d_bgr_, w, h, stream_);
    } else {
        nitros_cuda_fallback:
        // RGB8 → BGR8 in-place on a copy
        cudaMemcpyAsync(d_bgr_, d_src, frame_bytes, cudaMemcpyDeviceToDevice, stream_);
        cuda_rgb_to_bgr_inplace(d_bgr_, w, h, stream_);
    }

    AnnotateAndPublish(w, h, header);
}

// ─────────────────────────────────────────────────────────────────────────────
// AnnotateAndPublish — called by all input paths with d_bgr_ filled
// ─────────────────────────────────────────────────────────────────────────────
void AnnotatedImagePublisherNode::AnnotateAndPublish(
    int img_w, int img_h,
    const std_msgs::msg::Header & header)
{
    const int pitch = img_w * 3;

    // ── Fetch cached detections ───────────────────────────────────────────
    Detection2DArray::SharedPtr dets;
    {
        std::lock_guard<std::mutex> lk(det_mtx_);
        dets = last_det_;
    }

    // Bounding boxes are normalised [0, 1]; scale directly by the actual image dimensions.
    const float scale_x = static_cast<float>(img_w);
    const float scale_y = static_cast<float>(img_h);

    // ── Build BBoxDraw list ────────────────────────────────────────────────
    std::vector<BBoxDraw>       boxes;
    std::vector<std::string>    labels;
    std::vector<int>            label_x_host, label_y_host;

    if (dets) {
        for (const auto & det : dets->detections) {
            if (det.results.empty()) continue;

            // Best hypothesis
            // class_id may be a human-readable name (e.g. "person") or a
            // numeric string (e.g. "0") depending on whether the upstream
            // node loaded a class-names file.  Use stoi with a catch-all
            // fallback so neither case crashes us.
            float best_score = 0.f;
            int   best_cls   = 0;           // numeric class index for color
            std::string best_cls_str;       // raw string for label display
            for (const auto & hyp : det.results) {
                const float s = static_cast<float>(hyp.hypothesis.score);
                if (s > best_score) {
                    best_score   = s;
                    best_cls_str = hyp.hypothesis.class_id;
                    // Try to parse as integer (numeric id path)
                    try {
                        best_cls = std::stoi(hyp.hypothesis.class_id);
                    } catch (...) {
                        // class_id is a human-readable name (e.g. "person").
                        // Hash it to a stable integer so every unique class
                        // name maps to a distinct, consistent color regardless
                        // of the order detections arrive.
                        best_cls = static_cast<int>(
                            std::hash<std::string>{}(hyp.hypothesis.class_id)
                            % static_cast<size_t>(std::max(1, num_classes_)));
                    }
                }
            }
            if (best_score < confidence_threshold_) continue;

            // BBox (center format from detection msg)
            const auto & bb = det.bbox;
            const float cx = static_cast<float>(bb.center.position.x);
            const float cy = static_cast<float>(bb.center.position.y);
            const float bw = static_cast<float>(bb.size_x);
            const float bh = static_cast<float>(bb.size_y);

            const int x1 = std::max(0, static_cast<int>((cx - bw * 0.5f) * scale_x));
            const int y1 = std::max(0, static_cast<int>((cy - bh * 0.5f) * scale_y));
            const int x2 = std::min(img_w - 1,
                static_cast<int>((cx + bw * 0.5f) * scale_x));
            const int y2 = std::min(img_h - 1,
                static_cast<int>((cy + bh * 0.5f) * scale_y));
            if (x2 <= x1 || y2 <= y1) continue;

            uint8_t b_col, g_col, r_col;
            if (box_color_b_ < 0) {
                class_color(best_cls, num_classes_, b_col, g_col, r_col);
            } else {
                b_col = static_cast<uint8_t>(box_color_b_);
                g_col = static_cast<uint8_t>(box_color_g_);
                r_col = static_cast<uint8_t>(box_color_r_);
            }

            BBoxDraw bd;
            bd.x1 = x1; bd.y1 = y1; bd.x2 = x2; bd.y2 = y2;
            bd.b  = b_col; bd.g = g_col; bd.r = r_col;
            bd.thickness = box_thickness_;
            boxes.push_back(bd);

            // Label string — use the raw class_id string (name or number)
            if (draw_labels_ || draw_confidence_) {
                std::ostringstream oss;
                if (draw_labels_)     oss << best_cls_str;
                if (draw_confidence_) {
                    if (draw_labels_) oss << ' ';
                    oss << static_cast<int>(best_score * 100) << '%';
                }
                labels.push_back(oss.str());
                label_x_host.push_back(x1 + 2);
                label_y_host.push_back(std::max(0, y1 - glyph_h_));
            }
        }
    }

    // ── Draw bounding boxes (GPU) ─────────────────────────────────────────
    if (!boxes.empty()) {
        annotation_draw_rects_gpu(d_bgr_, img_w, img_h, pitch,
            boxes.data(), static_cast<int>(boxes.size()), stream_);
    }

    // ── Draw text labels (GPU) ────────────────────────────────────────────
    if (!labels.empty() && atlas_tex_ != 0) {
        // Build one filled background rect per label (same colour as box, 1px padding)
        std::vector<BBoxDraw> bg_rects;
        bg_rects.reserve(labels.size());
        for (size_t i = 0; i < labels.size(); ++i) {
            const int text_px_w = static_cast<int>(labels[i].size()) * glyph_w_ + 2;
            const int bg_x1 = std::max(0, label_x_host[i] - 1);
            const int bg_y1 = std::max(0, label_y_host[i] - 1);
            const int bg_x2 = std::min(img_w - 1, bg_x1 + text_px_w);
            const int bg_y2 = std::min(img_h - 1, bg_y1 + glyph_h_ + 2);
            // Use the same colour as the corresponding bounding box
            const BBoxDraw & src = boxes[i];
            BBoxDraw bg;
            bg.x1 = bg_x1; bg.y1 = bg_y1; bg.x2 = bg_x2; bg.y2 = bg_y2;
            // Darken slightly (halve intensity) so white text is clearly visible
            bg.b  = static_cast<uint8_t>(src.b >> 1);
            bg.g  = static_cast<uint8_t>(src.g >> 1);
            bg.r  = static_cast<uint8_t>(src.r >> 1);
            bg.thickness = 1;
            bg_rects.push_back(bg);
        }
        annotation_draw_fill_rects_gpu(d_bgr_, img_w, img_h, pitch,
            bg_rects.data(), static_cast<int>(bg_rects.size()), stream_);

        // Pack label chars flat
        std::string flat_chars;
        std::vector<int> h_offsets, h_lens;
        for (const auto & lbl : labels) {
            h_offsets.push_back(static_cast<int>(flat_chars.size()));
            h_lens.push_back(static_cast<int>(lbl.size()));
            flat_chars += lbl;
        }
        const int n_labels = static_cast<int>(labels.size());
        EnsureLabelDevBufs(n_labels, static_cast<int>(flat_chars.size()));

        cudaMemcpyAsync(d_label_chars_, flat_chars.data(), flat_chars.size(),
            cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_label_offsets_, h_offsets.data(), n_labels * sizeof(int),
            cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_label_lens_, h_lens.data(), n_labels * sizeof(int),
            cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_label_x_, label_x_host.data(), n_labels * sizeof(int),
            cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_label_y_, label_y_host.data(), n_labels * sizeof(int),
            cudaMemcpyHostToDevice, stream_);

        annotation_draw_text_gpu(
            d_bgr_, img_w, img_h, pitch,
            d_label_chars_, d_label_offsets_, d_label_lens_,
            d_label_x_, d_label_y_,
            255, 255, 255,   // white text on darkened box-colour background
            n_labels,
            atlas_tex_, glyph_w_, glyph_h_,
            stream_);
    }

    // ── Synchronize stream ────────────────────────────────────────────────
    cudaStreamSynchronize(stream_);

    // ── Publish annotated image ───────────────────────────────────────────
    const size_t frame_bytes = static_cast<size_t>(img_w) * img_h * 3;

    if (pub_transport_ == PubTransport::COMPRESSED) {
        CompressedMsg out;
        out.header  = header;
        out.format  = "jpeg";
        bool encoded = false;
#ifdef WITH_NVJPEG
        if (nvjpeg_handle_ && enc_state_ && enc_params_) {
            nvjpegImage_t nv_img{};
            // nvJPEG encode expects planar BGR; we have packed BGR.
            // Use the simplest approach: encode packed via NVJPEG_INPUT_BGR.
            nv_img.channel[0] = d_bgr_;
            nv_img.pitch[0]   = static_cast<unsigned int>(pitch);

            if (nvjpegEncodeImage(nvjpeg_handle_, enc_state_, enc_params_,
                    &nv_img, NVJPEG_INPUT_BGR, img_w, img_h, stream_)
                == NVJPEG_STATUS_SUCCESS)
            {
                size_t len = 0;
                nvjpegEncodeRetrieveBitstream(nvjpeg_handle_, enc_state_, nullptr, &len, stream_);
                cudaStreamSynchronize(stream_);
                out.data.resize(len);
                nvjpegEncodeRetrieveBitstream(nvjpeg_handle_, enc_state_,
                    out.data.data(), &len, stream_);
                cudaStreamSynchronize(stream_);
                encoded = true;
            }
        }
#endif
        if (!encoded) {
            // CPU fallback: DtoH → cv::imencode
            EnsureStagingBuf(frame_bytes);
            cudaMemcpy(h_staging_, d_bgr_, frame_bytes, cudaMemcpyDeviceToHost);
            cv::Mat bgr_mat(img_h, img_w, CV_8UC3, h_staging_);
            std::vector<uint8_t> jpeg_buf;
            cv::imencode(".jpg", bgr_mat, jpeg_buf,
                {cv::IMWRITE_JPEG_QUALITY, 85});
            out.data = std::move(jpeg_buf);
        }
        comp_pub_->publish(out);
    } else {
        // RAW publish: DtoH → sensor_msgs/Image bgr8
        EnsureStagingBuf(frame_bytes);
        cudaMemcpy(h_staging_, d_bgr_, frame_bytes, cudaMemcpyDeviceToHost);
        ImageMsg out;
        out.header   = header;
        out.encoding = "bgr8";
        out.width    = static_cast<uint32_t>(img_w);
        out.height   = static_cast<uint32_t>(img_h);
        out.step     = static_cast<uint32_t>(pitch);
        out.is_bigendian = false;
        out.data.resize(frame_bytes);
        std::memcpy(out.data.data(), h_staging_, frame_bytes);
        raw_pub_->publish(out);
    }

    // ── Re-publish CameraInfo ─────────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(cam_info_mtx_);
        if (cached_cam_info_) {
            auto ci = *cached_cam_info_;  // copy
            ci.header = header;           // update stamp
            cam_info_pub_->publish(ci);
        }
    }
}

}}}  // namespace nvidia::isaac_ros::object_detection

RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::object_detection::AnnotatedImagePublisherNode)

