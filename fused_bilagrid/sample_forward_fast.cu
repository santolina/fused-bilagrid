#include "config.h"

// Optimized forward kernel with shared memory caching for spatial locality
__global__ void bilagrid_sample_forward_kernel_fast(
    const float* __restrict__ bilagrid, // [N,12,L,H,W]
    const float* __restrict__ coords,  // [N,m,h,w,2]
    const float* __restrict__ rgb,  // [N,m,h,w,3]
    float* __restrict__ output,  // [N,m,h,w,3]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    // Shared memory for caching bilagrid data
    // Cache size optimized for spatial locality
    __shared__ float cache[12][2][4][4]; // [channels][z_levels][y_range][x_range]
    
    int block_w = 16; // block width for spatial locality
    int block_h = 16; // block height for spatial locality
    
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;
    
    if (global_x >= w || global_y >= h || batch_idx >= N * m) return;
    
    int mi = batch_idx % m;
    int ni = batch_idx / m;
    
    // Load sample data
    int g_offset = (((ni * m + mi) * h + global_y) * w + global_x);
    float sr = rgb[3*g_offset+0];
    float sg = rgb[3*g_offset+1];
    float sb = rgb[3*g_offset+2];
    
    // Compute grid coordinates
    float gx = coords[2*g_offset+0];
    float gy = coords[2*g_offset+1];
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
    
    // Interpolation parameters
    float fx = x - (float)x0;
    float fy = y - (float)y0;
    float fz = z - (float)z0;
    
    // Check if we can use cached data (spatial locality optimization)
    bool use_cache = (x0 >= blockIdx.x * blockDim.x - 1) && 
                     (x1 <= blockIdx.x * blockDim.x + blockDim.x) &&
                     (y0 >= blockIdx.y * blockDim.y - 1) && 
                     (y1 <= blockIdx.y * blockDim.y + blockDim.y);
    
    float dr = 0.0, dg = 0.0, db = 0.0;
    
    if (use_cache) {
        // Use shared memory cache for better memory coalescing
        __syncthreads();
        
        // Collaborative loading into shared memory
        if (local_x < 4 && local_y < 4) {
            int cache_x = local_x;
            int cache_y = local_y;
            int grid_x = min(blockIdx.x * blockDim.x + cache_x, W-1);
            int grid_y = min(blockIdx.y * blockDim.y + cache_y, H-1);
            
            for (int ci = 0; ci < 12; ci++) {
                const float* vol = &bilagrid[((ni*12 + ci)*L*H*W)];
                cache[ci][0][cache_y][cache_x] = vol[(z0*H+grid_y)*W+grid_x];
                cache[ci][1][cache_y][cache_x] = vol[(z1*H+grid_y)*W+grid_x];
            }
        }
        __syncthreads();
        
        // Use cached data for interpolation
        for (int ci = 0; ci < 12; ci++) {
            int rel_x0 = x0 - blockIdx.x * blockDim.x;
            int rel_y0 = y0 - blockIdx.y * blockDim.y;
            int rel_x1 = x1 - blockIdx.x * blockDim.x;
            int rel_y1 = y1 - blockIdx.y * blockDim.y;
            
            if (rel_x0 >= 0 && rel_x1 < 4 && rel_y0 >= 0 && rel_y1 < 4) {
                float v000 = cache[ci][0][rel_y0][rel_x0];
                float v001 = cache[ci][0][rel_y0][rel_x1];
                float v010 = cache[ci][0][rel_y1][rel_x0];
                float v011 = cache[ci][0][rel_y1][rel_x1];
                float v100 = cache[ci][1][rel_y0][rel_x0];
                float v101 = cache[ci][1][rel_y0][rel_x1];
                float v110 = cache[ci][1][rel_y1][rel_x0];
                float v111 = cache[ci][1][rel_y1][rel_x1];
                
                // Trilinear interpolation
                float c00 = v000*(1.0f-fx) + v001*fx;
                float c01 = v010*(1.0f-fx) + v011*fx;
                float c10 = v100*(1.0f-fx) + v101*fx;
                float c11 = v110*(1.0f-fx) + v111*fx;
                float c0 = c00*(1.0f-fy) + c01*fy;
                float c1 = c10*(1.0f-fy) + c11*fy;
                float val = c0*(1.0f-fz) + c1*fz;
                
                // Affine transform
                int si = ci % 4;
                int di = ci / 4;
                (di == 0 ? dr : di == 1 ? dg : db) += val * 
                    (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.0f);
            }
        }
    } else {
        // Fall back to original implementation for boundary cases
        #pragma unroll
        for (int ci = 0; ci < 12; ci++) {
            const float* vol = &bilagrid[((ni*12 + ci)*L*H*W)];
            
            auto v000 = vol[(z0*H+y0)*W+x0];
            auto v001 = vol[(z0*H+y0)*W+x1];
            auto v010 = vol[(z0*H+y1)*W+x0];
            auto v011 = vol[(z0*H+y1)*W+x1];
            auto v100 = vol[(z1*H+y0)*W+x0];
            auto v101 = vol[(z1*H+y0)*W+x1];
            auto v110 = vol[(z1*H+y1)*W+x0];
            auto v111 = vol[(z1*H+y1)*W+x1];
            
            float c00 = v000*(1.0f-fx) + v001*fx;
            float c01 = v010*(1.0f-fx) + v011*fx;
            float c10 = v100*(1.0f-fx) + v101*fx;
            float c11 = v110*(1.0f-fx) + v111*fx;
            float c0 = c00*(1.0f-fy) + c01*fy;
            float c1 = c10*(1.0f-fy) + c11*fy;
            float val = c0*(1.0f-fz) + c1*fz;
            
            int si = ci % 4;
            int di = ci / 4;
            (di == 0 ? dr : di == 1 ? dg : db) += val * 
                (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.0f);
        }
    }
    
    output[3*g_offset+0] = dr;
    output[3*g_offset+1] = dg;
    output[3*g_offset+2] = db;
}

void bilagrid_sample_forward_fast(
    const float* bilagrid,
    const float* coords,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w
) {
    // Optimized block configuration for spatial locality
    dim3 block(16, 16, 1);
    dim3 grid((w + block.x - 1) / block.x, 
              (h + block.y - 1) / block.y, 
              N * m);
    
    bilagrid_sample_forward_kernel_fast<<<grid, block>>>(
        bilagrid, coords, rgb, output,
        N, L, H, W, m, h, w
    );
}
