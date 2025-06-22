#include "config.h"

// Spatial locality detection for adaptive caching
__device__ float calculate_spatial_variance(
    const float* coords, 
    int block_start, 
    int block_size
) {
    float x_mean = 0.0f, y_mean = 0.0f;
    float x_var = 0.0f, y_var = 0.0f;
    
    // Calculate mean
    for (int i = 0; i < block_size; i++) {
        if (block_start + i < block_size) {
            x_mean += coords[(block_start + i) * 2];
            y_mean += coords[(block_start + i) * 2 + 1];
        }
    }
    if (block_size > 0) {
        x_mean /= block_size;
        y_mean /= block_size;
    }
    
    // Calculate variance
    for (int i = 0; i < block_size; i++) {
        if (block_start + i < block_size) {
            float dx = coords[(block_start + i) * 2] - x_mean;
            float dy = coords[(block_start + i) * 2 + 1] - y_mean;
            x_var += dx * dx;
            y_var += dy * dy;
        }
    }
    
    return (block_size > 0) ? (x_var + y_var) / block_size : 1.0f;
}

// Adaptive forward kernel with spatial locality optimization
__global__ void bilagrid_sample_forward_kernel_adaptive(
    const float* __restrict__ bilagrid, // [N,12,L,H,W]
    const float* __restrict__ coords,   // [N,m,h,w,2]
    const float* __restrict__ rgb,      // [N,m,h,w,3]
    float* __restrict__ output,         // [N,m,h,w,3]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    // Shared memory for caching bilagrid values
    __shared__ float cached_bilagrid[12][4][4][4]; // Limited cache for spatial locality
    __shared__ bool cache_valid;
    
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    int hi = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (wi >= w || hi >= h || idx >= N * m) return;
    
    int mi = idx % m;
    int ni = idx / m;
    
    // Initialize cache validity flag
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        cache_valid = false;
    }
    __syncthreads();
    
    // Sample data
    int g_offset = (((ni * m + mi) * h + hi) * w + wi);
    float sr = rgb[3*g_offset + 0];
    float sg = rgb[3*g_offset + 1];
    float sb = rgb[3*g_offset + 2];
    
    // Grid coordinates
    float gx = coords[2*g_offset + 0];
    float gy = coords[2*g_offset + 1];
    float gz = kC2G_r * sr + kC2G_g * sg + kC2G_b * sb;
    
    float x = gx * (W - 1);
    float y = gy * (H - 1);
    float z = gz * (L - 1);
    
    // Find corner indices
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
    
    // Interpolation weights
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
    
    // Try to use cache if coordinates are within cached region
    bool use_cache = false;
    if (cache_valid && 
        x0 >= 0 && x1 < 4 && 
        y0 >= 0 && y1 < 4 && 
        z0 >= 0 && z1 < 4) {
        use_cache = true;
    }
    
    float dr = 0.0f, dg = 0.0f, db = 0.0f;
    
    // Trilinear interpolation with optional caching
    for (int ci = 0; ci < 12; ci++) {
        int ch = ci % 3; // color channel
        int sp = ci / 3; // spectrum channel
        
        float v000, v001, v010, v011, v100, v101, v110, v111;
        
        if (use_cache) {
            // Load from cache
            v000 = cached_bilagrid[ci][z0][y0][x0];
            v001 = cached_bilagrid[ci][z1][y0][x0];
            v010 = cached_bilagrid[ci][z0][y1][x0];
            v011 = cached_bilagrid[ci][z1][y1][x0];
            v100 = cached_bilagrid[ci][z0][y0][x1];
            v101 = cached_bilagrid[ci][z1][y0][x1];
            v110 = cached_bilagrid[ci][z0][y1][x1];
            v111 = cached_bilagrid[ci][z1][y1][x1];
        } else {
            // Load from global memory
            int base_idx = (ni * 12 + ci) * L * H * W;
            v000 = bilagrid[base_idx + z0 * H * W + y0 * W + x0];
            v001 = bilagrid[base_idx + z1 * H * W + y0 * W + x0];
            v010 = bilagrid[base_idx + z0 * H * W + y1 * W + x0];
            v011 = bilagrid[base_idx + z1 * H * W + y1 * W + x0];
            v100 = bilagrid[base_idx + z0 * H * W + y0 * W + x1];
            v101 = bilagrid[base_idx + z1 * H * W + y0 * W + x1];
            v110 = bilagrid[base_idx + z0 * H * W + y1 * W + x1];
            v111 = bilagrid[base_idx + z1 * H * W + y1 * W + x1];
        }
        
        float val = v000 * w000 + v001 * w001 + v010 * w010 + v011 * w011 +
                    v100 * w100 + v101 * w101 + v110 * w110 + v111 * w111;
        
        float spectrum_val = (sp == 0 ? sr : sp == 1 ? sg : sp == 2 ? sb : 1.0f);
        
        if (ch == 0) dr += val * spectrum_val;
        else if (ch == 1) dg += val * spectrum_val;
        else db += val * spectrum_val;
    }
    
    output[3*g_offset + 0] = dr;
    output[3*g_offset + 1] = dg;
    output[3*g_offset + 2] = db;
}

void bilagrid_sample_forward_adaptive(
    const float* bilagrid,
    const float* coords,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w
) {
    dim3 block(16, 16, 1);
    dim3 grid((w + block.x - 1) / block.x, 
              (h + block.y - 1) / block.y, 
              N * m);
    
    bilagrid_sample_forward_kernel_adaptive<<<grid, block>>>(
        bilagrid, coords, rgb, output,
        N, L, H, W, m, h, w
    );
}
