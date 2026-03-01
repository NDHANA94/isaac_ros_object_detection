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
 * @file: annotation_kernels.hpp
 * @brief: Shared types and C++ API declarations for the CUDA annotation kernels
 *         implemented in annotation_kernels.cu.
 * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
 * @company: Sintez.LLC
 * @date: 2026-03-01
 * 
 * This header declares the C++ API for GPU-accelerated annotation drawing
 * kernels, as well as some color-format conversion helpers.  
 * 
 * All functions are thin C++ wrappers around CUDA kernels implemented in 
 * annotation_kernels.cu, so that the main node implementation can call them 
 * without needing to know any CUDA details.  The annotation_kernels.cu file 
 * also contains the actual CUDA kernel implementations, which are kept separate 
 * to maintain a clean separation between the node logic and the GPU code.  
 * 
 * This header is included by annotated_image_publisher_node.cpp and 
 * object_detection_node.cpp, which call the annotation drawing functions to 
 * render bounding boxes and text labels on the GPU before publishing annotated 
 * images.  
 * 
 * The color-format helper functions are used to convert between RGB and BGR 
 * formats as needed, since NITROS may deliver RGB8 buffers but the annotation 
 * kernels expect BGR8.
 * ─────────────────────────────────────────────────────────────────────────────
*/


#include <cuda_runtime.h>
#include <stdint.h>

// ─────────────────────────────────────────────────────────────────────────────
// BBoxDraw  — plain-old-data descriptor for a single annotated bounding box.
// ─────────────────────────────────────────────────────────────────────────────
struct BBoxDraw
{
    int     x1, y1, x2, y2;   //< pixel-space bounding box (clamped to image)
    uint8_t b, g, r;           //< BGR colour of the box border/fill
    int     thickness;         //< border width in pixels
};

// ─────────────────────────────────────────────────────────────────────────────
// Public API (defined in annotation_kernels.cu)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Draw bounding-box outlines on a device-resident packed BGR (BGRI) image.
 *
 * @param d_img      Device pointer to the packed-BGR buffer (3 bytes/pixel).
 * @param width      Image width  in pixels.
 * @param height     Image height in pixels.
 * @param pitch      Row stride in bytes (nvJPEG pitch[0]).
 * @param h_boxes    Host array of BBoxDraw descriptors (copied to device internally).
 * @param num_boxes  Number of elements in h_boxes.
 * @param stream     CUDA stream on which all work is queued.
 * @return true on success, false on any CUDA error.
 */
bool annotation_draw_rects_gpu(
    uint8_t*        d_img,
    int             width,
    int             height,
    int             pitch,
    const BBoxDraw* h_boxes,
    int             num_boxes,
    cudaStream_t    stream);

/**
 * Draw filled rectangles on a device-resident packed BGR image.
 * Typically used to paint opaque label-background banners.
 */
bool annotation_draw_fill_rects_gpu(
    uint8_t*        d_img,
    int             width,
    int             height,
    int             pitch,
    const BBoxDraw* h_boxes,
    int             num_boxes,
    cudaStream_t    stream);

// ─────────────────────────────────────────────────────────────────────────────
// GPU text rendering via pre-built glyph atlas
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Stamp text labels onto a device-resident packed BGR image using a
 * pre-uploaded glyph atlas texture.
 *
 * @param d_img          Device pointer to packed-BGR buffer (3 bytes/pixel).
 * @param img_w          Image width in pixels.
 * @param img_h          Image height in pixels.
 * @param pitch          Row stride in bytes (== img_w * 3 for tightly packed).
 * @param d_labels_buf   Flat device buffer of all label chars (NOT null-terminated).
 * @param d_offsets      Per-label byte offset into d_labels_buf  [num_labels].
 * @param d_lens         Per-label character count                [num_labels].
 * @param d_x            Per-label top-left x position            [num_labels].
 * @param d_y            Per-label top-left y position            [num_labels].
 * @param text_b/g/r     BGR foreground colour for text glyphs.
 * @param num_labels     Number of labels to draw.
 * @param atlas_tex      CUDA texture object for the glyph atlas (uint8, 2D).
 * @param glyph_w        Width  of each glyph cell in the atlas (pixels).
 * @param glyph_h        Height of each glyph cell in the atlas (pixels).
 * @param stream         CUDA stream.
 * @return true on success.
 */
bool annotation_draw_text_gpu(
    uint8_t*              d_img,
    int                   img_w,
    int                   img_h,
    int                   pitch,
    const uint8_t*        d_labels_buf,
    const int*            d_offsets,
    const int*            d_lens,
    const int*            d_x,
    const int*            d_y,
    uint8_t               text_b,
    uint8_t               text_g,
    uint8_t               text_r,
    int                   num_labels,
    cudaTextureObject_t   atlas_tex,
    int                   glyph_w,
    int                   glyph_h,
    cudaStream_t          stream);

// ─────────────────────────────────────────────────────────────────────────────
// Color-format helper kernels (thin C entry points, implemented in
// annotation_kernels.cu so they live near the rest of the annotation code)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Swap R and B channels in-place for a packed RGB/BGR device buffer.
 * Used when NITROS delivers an RGB8 buffer and we need BGR.
 */
void cuda_rgb_to_bgr_inplace(
    uint8_t*     d_buf,
    int          width,
    int          height,
    cudaStream_t stream);

/**
 * Convert packed RGBA → packed BGR (drop alpha).
 * Used after VIC outputs an RGBA NvBufSurface.
 */
void cuda_rgba_to_bgr_packed(
    const uint8_t* d_rgba,
    uint8_t*       d_bgr,
    int            width,
    int            height,
    cudaStream_t   stream);

/**
 * Interleave three separate uint8 device planes (B, G, R) into a
 * tightly packed BGR buffer.  Used after nvJPEG decompression which
 * returns planar output.
 */
void cuda_planar_to_packed_bgr(
    const uint8_t* d_b,
    const uint8_t* d_g,
    const uint8_t* d_r,
    uint8_t*       d_bgr,
    int            width,
    int            height,
    cudaStream_t   stream);
