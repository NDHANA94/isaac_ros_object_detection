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
 * @file: annotation_kernels.cu
 * @brief: CUDA kernels for image annotation (bounding boxes, text).
 * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
 * @company: Sintez.LLC
 * @date: 2026-03-01
 * 
 * This file implements CUDA kernels for drawing bounding boxes and text annotations
 * directly on images in GPU memory.  These kernels are designed to be efficient
 * for real-time applications, minimizing data transfer between host and device.
 * 
 * All image-annotation work runs entirely on the GPU:
 *   • Bounding-box border / filled-rect drawing
 *   • Glyph-atlas text stamping
 *   • Color-format conversion helpers (rgb↔bgr, rgba→bgr, planar→packed)
 * 
 * Input/output buffers are ALWAYS packed BGR (3 bytes/pixel, row-major) unless
 * stated otherwise.  All kernels are queued on a caller-supplied cudaStream.
 * ─────────────────────────────────────────────────────────────────────────────
*/


#include "isaac_ros_object_detection/annotation_kernels.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// Macro helpers
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "[annotation_kernels] CUDA error: %s  at %s:%d\n",      \
              cudaGetErrorString(_e), __FILE__, __LINE__);                     \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_VOID(call)                                                  \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess)                                                     \
      fprintf(stderr, "[annotation_kernels] CUDA error: %s  at %s:%d\n",      \
              cudaGetErrorString(_e), __FILE__, __LINE__);                     \
  } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Bounding-box border drawing
//
//  Strategy: each thread is assigned to ONE absolute pixel (px, py) and tests
//  whether that pixel falls within the border of ANY box.  This is O(num_boxes)
//  per thread but num_boxes is small (< 300) and the branching is uniform
//  across a warp when boxes don't overlap.
//
//  A more cache-friendly alternative would be to launch one block per box edge,
//  but the simple approach is easier to reason about and fast enough for < 100
//  detections at 1280x480@30fps on Orin.
// ─────────────────────────────────────────────────────────────────────────────

__constant__ BBoxDraw c_boxes[300];   // ← reused by rect and fill kernels

__global__ void k_draw_rects(
    uint8_t* __restrict__ img,
    int width, int height, int pitch,
    int num_boxes)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    for (int i = 0; i < num_boxes; ++i) {
        const BBoxDraw& b = c_boxes[i];
        const int t = b.thickness;

        // Check if (px,py) is in any of the four border bands
        const bool in_top    = (py >= b.y1 && py < b.y1 + t  && px >= b.x1 && px <= b.x2);
        const bool in_bottom = (py >  b.y2 - t && py <= b.y2  && px >= b.x1 && px <= b.x2);
        const bool in_left   = (px >= b.x1 && px < b.x1 + t  && py >= b.y1 && py <= b.y2);
        const bool in_right  = (px >  b.x2 - t && px <= b.x2  && py >= b.y1 && py <= b.y2);

        if (in_top || in_bottom || in_left || in_right) {
            const int idx = py * pitch + px * 3;
            img[idx + 0] = b.b;
            img[idx + 1] = b.g;
            img[idx + 2] = b.r;
        }
    }
}

bool annotation_draw_rects_gpu(
    uint8_t*        d_img,
    int             width,
    int             height,
    int             pitch,
    const BBoxDraw* h_boxes,
    int             num_boxes,
    cudaStream_t    stream)
{
    if (num_boxes <= 0) return true;
    if (num_boxes > 300) num_boxes = 300;

    CUDA_CHECK(cudaMemcpyToSymbolAsync(
        c_boxes, h_boxes, num_boxes * sizeof(BBoxDraw), 0,
        cudaMemcpyHostToDevice, stream));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    k_draw_rects<<<grid, block, 0, stream>>>(d_img, width, height, pitch, num_boxes);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[annotation_kernels] k_draw_rects: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Filled rectangle drawing (label background banners)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_fill_rects(
    uint8_t* __restrict__ img,
    int width, int height, int pitch,
    int num_boxes)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    for (int i = 0; i < num_boxes; ++i) {
        const BBoxDraw& b = c_boxes[i];
        if (px >= b.x1 && px <= b.x2 && py >= b.y1 && py <= b.y2) {
            const int idx = py * pitch + px * 3;
            img[idx + 0] = b.b;
            img[idx + 1] = b.g;
            img[idx + 2] = b.r;
        }
    }
}

bool annotation_draw_fill_rects_gpu(
    uint8_t*        d_img,
    int             width,
    int             height,
    int             pitch,
    const BBoxDraw* h_boxes,
    int             num_boxes,
    cudaStream_t    stream)
{
    if (num_boxes <= 0) return true;
    if (num_boxes > 300) num_boxes = 300;

    CUDA_CHECK(cudaMemcpyToSymbolAsync(
        c_boxes, h_boxes, num_boxes * sizeof(BBoxDraw), 0,
        cudaMemcpyHostToDevice, stream));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    k_fill_rects<<<grid, block, 0, stream>>>(d_img, width, height, pitch, num_boxes);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[annotation_kernels] k_fill_rects: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Glyph-atlas text stamping
//
//  Grid: (glyph_col, glyph_row, label_index)
//    • threadIdx.x / blockIdx.x → column pixel within a glyph cell
//    • threadIdx.y / blockIdx.y → row pixel within a glyph cell
//    • blockIdx.z               → label index
//
//  For each glyph (char index within the label string), the thread at column ≡
//  glyph_col % glyph_w samples the atlas texture and writes the foreground
//  colour if the glyph pixel is set (> 128).
//
//  The atlas texture is single-channel uint8, width = 95 * glyph_w,
//  height = glyph_h, unnormalised integer coordinates.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_draw_text(
    uint8_t* __restrict__ img,
    int img_w, int img_h, int pitch,
    const uint8_t* __restrict__ labels_buf,   // flat packed chars (no null)
    const int*     __restrict__ offsets,      // [num_labels] byte offset in labels_buf
    const int*     __restrict__ lens,         // [num_labels] char count
    const int*     __restrict__ label_x,      // [num_labels] top-left x
    const int*     __restrict__ label_y,      // [num_labels] top-left y
    uint8_t text_b, uint8_t text_g, uint8_t text_r,
    cudaTextureObject_t atlas,
    int glyph_w, int glyph_h)
{
    // Each block handles one label; threads tile over the label's pixel area.
    const int label_idx = blockIdx.z;
    const int char_col  = blockIdx.x * blockDim.x + threadIdx.x;  // pixel col in label bitmap
    const int row       = blockIdx.y * blockDim.y + threadIdx.y;  // pixel row in glyph

    if (row >= glyph_h) return;

    const int num_chars = lens[label_idx];
    const int glyph_idx = char_col / glyph_w;   // which character in the string
    const int glyph_col = char_col % glyph_w;   // pixel col within that glyph cell

    if (glyph_idx >= num_chars) return;

    // Fetch the character and map to atlas column
    const uint8_t ch        = labels_buf[offsets[label_idx] + glyph_idx];
    const int     atlas_col = (ch >= 32 && ch <= 126)
                              ? (ch - 32) * glyph_w + glyph_col
                              : 0;  // unknown chars → blank (space is index 0)

    // Sample atlas (unnormalised coords)
    const uint8_t pixel = tex2D<uint8_t>(atlas, atlas_col, row);
    if (pixel < 128) return;   // background pixel → skip

    // Destination pixel in the image
    const int img_x = label_x[label_idx] + char_col;
    const int img_y = label_y[label_idx] + row;
    if (img_x < 0 || img_x >= img_w || img_y < 0 || img_y >= img_h) return;

    const int dst = img_y * pitch + img_x * 3;
    img[dst + 0] = text_b;
    img[dst + 1] = text_g;
    img[dst + 2] = text_r;
}

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
    cudaStream_t          stream)
{
    if (num_labels <= 0 || atlas_tex == 0) return true;

    // Conservative max label width: 64 chars * glyph_w pixels wide
    const int max_label_pixel_w = 64 * glyph_w;

    dim3 block(16, 8);
    dim3 grid(
        (max_label_pixel_w + block.x - 1) / block.x,
        (glyph_h           + block.y - 1) / block.y,
        static_cast<unsigned>(num_labels));

    k_draw_text<<<grid, block, 0, stream>>>(
        d_img, img_w, img_h, pitch,
        d_labels_buf, d_offsets, d_lens, d_x, d_y,
        text_b, text_g, text_r,
        atlas_tex, glyph_w, glyph_h);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[annotation_kernels] k_draw_text: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Color-format helpers
// ─────────────────────────────────────────────────────────────────────────────

// ── 4a. In-place R↔B swap (packed RGB → BGR or BGR → RGB) ────────────────────

__global__ void k_rgb_to_bgr_inplace(uint8_t* __restrict__ buf, int n_pixels)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;
    const int base = i * 3;
    const uint8_t tmp = buf[base + 0];
    buf[base + 0]     = buf[base + 2];
    buf[base + 2]     = tmp;
}

void cuda_rgb_to_bgr_inplace(uint8_t* d_buf, int width, int height, cudaStream_t stream)
{
    const int n = width * height;
    const int block = 256;
    k_rgb_to_bgr_inplace<<<(n + block - 1) / block, block, 0, stream>>>(d_buf, n);
    CUDA_CHECK_VOID(cudaGetLastError());
}

// ── 4b. RGBA → packed BGR (drop alpha) ────────────────────────────────────────

__global__ void k_rgba_to_bgr_packed(
    const uint8_t* __restrict__ rgba,
    uint8_t*       __restrict__ bgr,
    int n_pixels)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;
    const int src = i * 4;
    const int dst = i * 3;
    bgr[dst + 0] = rgba[src + 2];   // B = rgba.B  (RGBA order: R=0,G=1,B=2,A=3)
    bgr[dst + 1] = rgba[src + 1];   // G
    bgr[dst + 2] = rgba[src + 0];   // R
}

void cuda_rgba_to_bgr_packed(
    const uint8_t* d_rgba, uint8_t* d_bgr,
    int width, int height, cudaStream_t stream)
{
    const int n = width * height;
    const int block = 256;
    k_rgba_to_bgr_packed<<<(n + block - 1) / block, block, 0, stream>>>(d_rgba, d_bgr, n);
    CUDA_CHECK_VOID(cudaGetLastError());
}

// ── 4c. Planar B/G/R uint8 → packed BGR (nvJPEG output → packed) ──────────────

__global__ void k_planar_to_packed_bgr(
    const uint8_t* __restrict__ pb,
    const uint8_t* __restrict__ pg,
    const uint8_t* __restrict__ pr,
    uint8_t*       __restrict__ bgr,
    int n_pixels)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;
    const int dst = i * 3;
    bgr[dst + 0] = pb[i];
    bgr[dst + 1] = pg[i];
    bgr[dst + 2] = pr[i];
}

void cuda_planar_to_packed_bgr(
    const uint8_t* d_b, const uint8_t* d_g, const uint8_t* d_r,
    uint8_t* d_bgr,
    int width, int height, cudaStream_t stream)
{
    const int n = width * height;
    const int block = 256;
    k_planar_to_packed_bgr<<<(n + block - 1) / block, block, 0, stream>>>(
        d_b, d_g, d_r, d_bgr, n);
    CUDA_CHECK_VOID(cudaGetLastError());
}
