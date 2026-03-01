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
 * @file: cuda_kernels.cu
 * @brief: CUDA kernels for image preprocessing and YOLO output decoding
 * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
 * @company: Sintez.LLC
 * @date: 2026-03-01
 * 
 * This file implements CUDA kernels for:
 * 1. Converting NV12 images to normalized float32 NCHW format (for VIC fallback)
 * 2. Converting RGBA uint8 images to normalized float32 NCHW format (for VIC outputs)
 * 3. Decoding YOLO model outputs into bounding boxes
 * ─────────────────────────────────────────────────────────────────────────────
*/


#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>   

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_KERNEL_CHECK()                                                   \
  do {                                                                        \
    cudaError_t e = cudaGetLastError();                                       \
    if (e != cudaSuccess)                                                     \
      fprintf(stderr, "[CUDA kernel] %s\n", cudaGetErrorString(e));          \
  } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// NV12 → float32 NCHW  (fallback when VIC is disabled)
// NV12 layout: Y plane [H×W] followed by interleaved UV plane [H/2 × W]
// Output: float32 NCHW, normalized [0, 1], resized to (dst_w × dst_h)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_nv12_to_float_nchw(
  const uint8_t * __restrict__ src_y,
  const uint8_t * __restrict__ src_uv,
  int src_w, int src_h,
  float       * __restrict__ dst,   // [1, 3, dst_h, dst_w]
  int dst_w, int dst_h)
{
  const int dx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst_w || dy >= dst_h) return;

  // Bilinear sample coordinates in source space
  const float sx = (dx + 0.5f) * src_w  / dst_w - 0.5f;
  const float sy = (dy + 0.5f) * src_h  / dst_h - 0.5f;
  const int   x0 = max(0, (int)sx),  x1 = min(src_w - 1, x0 + 1);
  const int   y0 = max(0, (int)sy),  y1 = min(src_h - 1, y0 + 1);
  const float fx = sx - x0,          fy = sy - y0;

  // Y sample
  const float Y =
    (1 - fy) * ((1 - fx) * src_y[y0 * src_w + x0] + fx * src_y[y0 * src_w + x1]) +
         fy  * ((1 - fx) * src_y[y1 * src_w + x0] + fx * src_y[y1 * src_w + x1]);

  // UV sample (half resolution)
  const int ux0 = max(0, x0 / 2), ux1 = min(src_w / 2 - 1, ux0 + 1);
  const int uy0 = max(0, y0 / 2), uy1 = min(src_h / 2 - 1, uy0 + 1);
  const float U_raw =
    (1 - fy) * ((1 - fx) * src_uv[uy0 * src_w + ux0 * 2]
              +      fx  * src_uv[uy0 * src_w + ux1 * 2]) +
         fy  * ((1 - fx) * src_uv[uy1 * src_w + ux0 * 2]
              +      fx  * src_uv[uy1 * src_w + ux1 * 2]);
  const float V_raw =
    (1 - fy) * ((1 - fx) * src_uv[uy0 * src_w + ux0 * 2 + 1]
              +      fx  * src_uv[uy0 * src_w + ux1 * 2 + 1]) +
         fy  * ((1 - fx) * src_uv[uy1 * src_w + ux0 * 2 + 1]
              +      fx  * src_uv[uy1 * src_w + ux1 * 2 + 1]);

  const float U = U_raw - 128.f;
  const float V = V_raw - 128.f;

  // BT.601 YUV → RGB
  const float R = Y                   + 1.402f   * V;
  const float G = Y - 0.344136f * U  - 0.714136f * V;
  const float B = Y + 1.772f    * U;

  const int pixel = dy * dst_w + dx;
  const int plane = dst_h * dst_w;

  dst[0 * plane + pixel] = __saturatef(R / 255.f);  // R channel
  dst[1 * plane + pixel] = __saturatef(G / 255.f);  // G channel
  dst[2 * plane + pixel] = __saturatef(B / 255.f);  // B channel
}

extern "C" void cuda_preprocess_nv12(
  const void * d_src, int src_w, int src_h,
  float * d_dst,      int dst_w, int dst_h,
  cudaStream_t stream)
{
  const uint8_t * y  = static_cast<const uint8_t *>(d_src);
  const uint8_t * uv = y + src_w * src_h;

  dim3 block(16, 16);
  dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
  k_nv12_to_float_nchw<<<grid, block, 0, stream>>>(
    y, uv, src_w, src_h, d_dst, dst_w, dst_h);
  CUDA_KERNEL_CHECK();
}

// ─────────────────────────────────────────────────────────────────────────────
// RGBA uint8 → float32 NCHW  (after VIC resize outputs RGBA)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_rgba_to_float_nchw(
  const uint8_t * __restrict__ src,  // RGBA interleaved, w×h
  float         * __restrict__ dst,  // NCHW [1, 3, h, w]
  int w, int h)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  const int src_idx = (y * w + x) * 4;
  const int plane   = h * w;
  const int px      = y * w + x;

  dst[0 * plane + px] = src[src_idx + 0] / 255.f;  // R
  dst[1 * plane + px] = src[src_idx + 1] / 255.f;  // G
  dst[2 * plane + px] = src[src_idx + 2] / 255.f;  // B
  // Alpha channel discarded
}

extern "C" void cuda_normalize_rgba(
  const void * d_rgba, float * d_dst,
  int w, int h,
  cudaStream_t stream)
{
  dim3 block(16, 16);
  dim3 grid((w + 15) / 16, (h + 15) / 16);
  k_rgba_to_float_nchw<<<grid, block, 0, stream>>>(
    static_cast<const uint8_t *>(d_rgba), d_dst, w, h);
  CUDA_KERNEL_CHECK();
}

// ─────────────────────────────────────────────────────────────────────────────
// YOLO output decoder
//
// Expects transposed export format (default for ultralytics ≥ 8.1):
//   d_raw shape: [1, (4 + num_classes), num_predictions]  — column-major predictions
//
// Each prediction column:  [cx, cy, w, h, cls0_score, cls1_score, ...]
//
// Outputs: d_boxes_out [MAX_DETS, 6] = {cx, cy, w, h, confidence, class_id}
//          d_count_out: number of written detections (atomic)
// ─────────────────────────────────────────────────────────────────────────────

#define MAX_DETS_GPU 300

__global__ void k_decode_yolo(
  const float * __restrict__ raw,    // [4 + C, N]
  int    N,
  int    C,
  float  conf_thresh,
  float * __restrict__ out_boxes,    // [MAX_DETS_GPU, 6]
  int   * __restrict__ out_count)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  // Find best class
  float best_score = 0.f;
  int   best_cls   = -1;
  for (int c = 0; c < C; ++c) {
    const float s = raw[(4 + c) * N + i];
    if (s > best_score) { best_score = s; best_cls = c; }
  }

  if (best_score < conf_thresh) return;

  const int slot = atomicAdd(out_count, 1);
  if (slot >= MAX_DETS_GPU) { atomicSub(out_count, 1); return; }

  float * b = out_boxes + slot * 6;
  b[0] = raw[0 * N + i];   // cx
  b[1] = raw[1 * N + i];   // cy
  b[2] = raw[2 * N + i];   // w
  b[3] = raw[3 * N + i];   // h
  b[4] = best_score;
  b[5] = static_cast<float>(best_cls);
}

extern "C" void cuda_decode_yolo_output(
  const float * d_raw,
  int           num_predictions,
  int           num_classes,
  float         conf_thresh,
  float *       d_boxes_out,
  int *         d_count_out,
  cudaStream_t  stream)
{
  const int block = 256;
  const int grid  = (num_predictions + block - 1) / block;
  k_decode_yolo<<<grid, block, 0, stream>>>(
    d_raw, num_predictions, num_classes, conf_thresh,
    d_boxes_out, d_count_out);
  CUDA_KERNEL_CHECK();
}