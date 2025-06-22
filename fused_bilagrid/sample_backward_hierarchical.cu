#include "config.h"

// Hierarchical reduction backward kernel (inspired by uniform optimizations)
__global__ void bilagrid_sample_backward_kernel_hierarchical(
    const float* __restrict__ bilagrid,  // [N,12,L,H,W]
    const float* __restrict__ coords,    // [N,m,h,w,2]
    const float* __restrict__ rgb,       // [N,m,h,w,3]
    const float* __restrict__ v_output,  // [N,m,h,w,3]
    float* __restrict__ v_bilagrid,      // [N,12,L,H,W]
    float* __restrict__ v_rgb,           // [N,m,h,w,3]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    // Shared memory for reducing atomic contention
    __shared__ float shared_accum[12][8]; // 12 channels x 8 corners
    
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    int hi = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (wi >= w || hi >= h || idx >= N * m) return;
    
    int mi = idx % m;
    int ni = idx / m;
    
    // Initialize shared memory
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    if (tid < 96) { // 12 * 8 = 96
        ((float*)shared_accum)[tid] = 0.0f;
    }
    __syncthreads();
    
    // Compute gradients
    int g_offset = (((ni * m + mi) * h + hi) * w + wi);
    
    float sr = rgb[3*g_offset + 0];
    float sg = rgb[3*g_offset + 1]; 
    float sb = rgb[3*g_offset + 2];
    
    float gx = coords[2*g_offset + 0];
    float gy = coords[2*g_offset + 1];
    float gz = kC2G_r * sr + kC2G_g * sg + kC2G_b * sb;
    
    float x = gx * (W - 1);
    float y = gy * (H - 1);
    float z = gz * (L - 1);
    
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int z0 = (int)floorf(z);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;
    
    x0 = min(max(x0, 0), W-1);
    x1 = min(max(x1, 0), W-1);
    y0 = min(max(y0, 0), H-1);
    y1 = min(max(y1, 0), H-1);
    z0 = min(max(z0, 0), L-1);
    z1 = min(max(z1, 0), L-1);
    
    float fx = x - (float)x0;
    float fy = y - (float)y0;
    float fz = z - (float)z0;
    
    float w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
    float w001 = (1.0f - fx) * (1.0f - fy) * fz;
    float w010 = (1.0f - fx) * fy * (1.0f - fz);
    float w011 = (1.0f - fx) * fy * fz;
    float w100 = fx * (1.0f - fy) * (1.0f - fz);
    float w101 = fx * (1.0f - fy) * fz;
    float w110 = fx * fy * (1.0f - fz);
    float w111 = fx * fy * fz;
    
    float v_r = v_output[3*g_offset + 0];
    float v_g = v_output[3*g_offset + 1];
    float v_b = v_output[3*g_offset + 2];
    
    // Accumulate to shared memory first to reduce atomic contention
    for (int ci = 0; ci < 12; ci++) {
        for (int corner = 0; corner < 8; corner++) {
            int ch = ci % 3;
            int sp = ci / 3;
            
            float w = 0.0f;
            int zc, yc, xc;
            
            if (corner == 0) { w = w000; zc = z0; yc = y0; xc = x0; }
            else if (corner == 1) { w = w001; zc = z1; yc = y0; xc = x0; }
            else if (corner == 2) { w = w010; zc = z0; yc = y1; xc = x0; }
            else if (corner == 3) { w = w011; zc = z1; yc = y1; xc = x0; }
            else if (corner == 4) { w = w100; zc = z0; yc = y0; xc = x1; }
            else if (corner == 5) { w = w101; zc = z1; yc = y0; xc = x1; }
            else if (corner == 6) { w = w110; zc = z0; yc = y1; xc = x1; }
            else if (corner == 7) { w = w111; zc = z1; yc = y1; xc = x1; }
            
            float v_val = (ch == 0 ? v_r : ch == 1 ? v_g : v_b);
            float sp_val = (sp == 0 ? sr : sp == 1 ? sg : sp == 2 ? sb : 1.0f);
            
            // Accumulate to shared memory
            atomicAdd(&shared_accum[ci][corner], w * v_val * sp_val);
        }
    }
    
    __syncthreads();
    
    // Write from shared memory to global memory (reduced atomic contention)
    if (tid < 96) {
        int ci = tid / 8;
        int corner = tid % 8;
        
        float accum_val = shared_accum[ci][corner];
        if (accum_val != 0.0f) {
            int zc, yc, xc;
            if (corner == 0) { zc = z0; yc = y0; xc = x0; }
            else if (corner == 1) { zc = z1; yc = y0; xc = x0; }
            else if (corner == 2) { zc = z0; yc = y1; xc = x0; }
            else if (corner == 3) { zc = z1; yc = y1; xc = x0; }
            else if (corner == 4) { zc = z0; yc = y0; xc = x1; }
            else if (corner == 5) { zc = z1; yc = y0; xc = x1; }
            else if (corner == 6) { zc = z0; yc = y1; xc = x1; }
            else if (corner == 7) { zc = z1; yc = y1; xc = x1; }
            
            int grid_idx = ((ni * 12 + ci) * L + zc) * H * W + yc * W + xc;
            atomicAdd(&v_bilagrid[grid_idx], accum_val);
        }
    }
    
    // RGB gradients
    v_rgb[3*g_offset + 0] = v_r;
    v_rgb[3*g_offset + 1] = v_g;
    v_rgb[3*g_offset + 2] = v_b;
}

void bilagrid_sample_backward_hierarchical(
    const float* bilagrid,
    const float* coords,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_coords,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w
) {
    dim3 block(16, 16, 1);
    dim3 grid((w + block.x - 1) / block.x, 
              (h + block.y - 1) / block.y, 
              (N * m + block.z - 1) / block.z);
    
    bilagrid_sample_backward_kernel_hierarchical<<<grid, block>>>(
        bilagrid, coords, rgb, v_output,
        v_bilagrid, v_rgb,
        N, L, H, W, m, h, w
    );
}
