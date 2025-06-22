#include "config.h"
// Note: cooperative_groups may require CUDA 11.0+ and compatible GPU architecture
#if __CUDACC_VER_MAJOR__ >= 11
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#define USE_COOPERATIVE_GROUPS 1
#else
#define USE_COOPERATIVE_GROUPS 0
#endif

// Optimized backward kernel with hierarchical reduction (ported from uniform version)
__global__ void bilagrid_sample_backward_kernel_hierarchical(
    const float* __restrict__ bilagrid,  // [N,12,L,H,W]
    const float* __restrict__ coords,  // [N,m,h,w,2]
    const float* __restrict__ rgb,  // [N,m,h,w,3]
    const float* __restrict__ v_output,  // [N,m,h,w,3]
    float* __restrict__ v_bilagrid,  // [N,12,L,H,W]
    float* __restrict__ v_rgb,  // [N,m,h,w,3]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    // Shared memory for hierarchical reduction (ported from uniform V1)
    __shared__ float local_accum[12][8]; // [channels][corners]
    __shared__ float reduction_buffer[256]; // For block-level reduction
    
    // Thread mapping optimized for atomicAdd reduction (from uniform V2)
    int wi = threadIdx.x * ((w+blockDim.x-1) / blockDim.x) + blockIdx.x;
    int hi = threadIdx.y * ((h+blockDim.y-1) / blockDim.y) + blockIdx.y;
    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    
    bool inside = (wi < w && hi < h && idx < (N*m));
    if (!inside) return;
    
    int mi = idx % m;
    int ni = idx / m;
    
    // Initialize shared memory
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    
    if (tid < 12 * 8) {
        ((float*)local_accum)[tid] = 0.0f;
    }
    __syncthreads();
    
    // Grid coordinates calculation
    int g_off = (((ni*m + mi)*h + hi)*w + wi);
    float sr = rgb[3*g_off+0], sg = rgb[3*g_off+1], sb = rgb[3*g_off+2];
    float gx = coords[2*g_off+0];
    float gy = coords[2*g_off+1];
    float gz = kC2G_r * sr + kC2G_g * sg + kC2G_b * sb;
    float x = gx * (W - 1);
    float y = gy * (H - 1);
    float z = gz * (L - 1);
    
    // Floor + ceil, clamped
    int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    x0 = min(max(x0,0), W-1); x1 = min(max(x1,0), W-1);
    y0 = min(max(y0,0), H-1); y1 = min(max(y1,0), H-1);
    z0 = min(max(z0,0), L-1); z1 = min(max(z1,0), L-1);
    
    // Fractional parts
    float fx = x - x0, fy = y - y0, fz = z - z0;
    float weights[8] = {
        (1-fx)*(1-fy)*(1-fz), fx*(1-fy)*(1-fz),
        (1-fx)*fy*(1-fz),     fx*fy*(1-fz),
        (1-fx)*(1-fy)*fz,     fx*(1-fy)*fz,
        (1-fx)*fy*fz,         fx*fy*fz
    };
    
    // Corner coordinates
    int corners_x[8] = {x0, x1, x0, x1, x0, x1, x0, x1};
    int corners_y[8] = {y0, y0, y1, y1, y0, y0, y1, y1};
    int corners_z[8] = {z0, z0, z0, z0, z1, z1, z1, z1};
    
    // Read upstream gradients
    float dr = v_output[3*g_off+0];
    float dg = v_output[3*g_off+1];
    float db = v_output[3*g_off+2];
    float vr = 0.0, vg = 0.0, vb = 0.0;
    
    // Phase 1: Local accumulation in shared memory (hierarchical reduction)
    #pragma unroll
    for (int ci = 0; ci < 12; ++ci) {
        int si = ci % 4, di = ci / 4;
        float r_coeff = (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.f);
        float gout = (di==0 ? dr : di==1 ? dg : db);
        float grad_weight = r_coeff * gout;
        
        // Accumulate to shared memory instead of direct atomicAdd
        #pragma unroll
        for (int corner = 0; corner < 8; ++corner) {
            atomicAdd(&local_accum[ci][corner], weights[corner] * grad_weight);
        }
        
        // RGB gradient calculation (unchanged)
        if (si < 3) {
            float val =
                ( ( (bilagrid[((ni*12 + ci)*L*H*W) + (z0*H + y0)*W + x0]*(1-fx) + 
                     bilagrid[((ni*12 + ci)*L*H*W) + (z0*H + y0)*W + x1]*fx)*(1-fy)
                    + (bilagrid[((ni*12 + ci)*L*H*W) + (z0*H + y1)*W + x0]*(1-fx) + 
                       bilagrid[((ni*12 + ci)*L*H*W) + (z0*H + y1)*W + x1]*fx)*fy )*(1-fz)
                + ( (bilagrid[((ni*12 + ci)*L*H*W) + (z1*H + y0)*W + x0]*(1-fx) + 
                     bilagrid[((ni*12 + ci)*L*H*W) + (z1*H + y0)*W + x1]*fx)*(1-fy)
                    + (bilagrid[((ni*12 + ci)*L*H*W) + (z1*H + y1)*W + x0]*(1-fx) + 
                       bilagrid[((ni*12 + ci)*L*H*W) + (z1*H + y1)*W + x1]*fx)*fy )*fz
                );
            (si == 0 ? vr : si == 1 ? vg : vb) += val * gout;
        }
    }
    
    __syncthreads();
    
    // Phase 2: Hierarchical reduction and global write (from uniform V1)
    if (tid < 8 * 12) { // 8 corners * 12 channels
        int corner = tid % 8;
        int channel = tid / 8;
        
        // Block-level reduction for this corner and channel
        reduction_buffer[tid] = local_accum[channel][corner];
        __syncthreads();
        
        // Tree reduction (binary reduction)
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockSize) {
                reduction_buffer[tid] += reduction_buffer[tid + s];
            }
            __syncthreads();
        }
        
        // Single thread writes to global memory (massive atomicAdd reduction)
        if (tid == 0 && local_accum[channel][corner] != 0.0f) {
            int xi = corners_x[corner];
            int yi = corners_y[corner]; 
            int zi = corners_z[corner];
            int base = ((ni*12 + channel)*L*H*W);
            int addr = base + (zi*H + yi)*W + xi;
            atomicAdd(v_bilagrid + addr, reduction_buffer[0]);
        }
    }
    
    // RGB gradient computation (unchanged)
    float dwdz[8] = {
        -(1-fx)*(1-fy), -fx*(1-fy),
        -(1-fx)*fy,     -fx*fy,
         (1-fx)*(1-fy),  fx*(1-fy),
         (1-fx)*fy,      fx*fy
    };
    
    float gz_grad = 0.f;
    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        int xi = corners_x[corner];
        int yi = corners_y[corner];
        int zi = corners_z[corner];
        float trilerp = 0.f;
        
        #pragma unroll
        for (int ci = 0; ci < 12; ++ci) {
            const float* vol = bilagrid + ((ni*12 + ci)*L*H*W);
            float v = vol[(zi*H + yi)*W + xi];
            int si = ci % 4, di = ci / 4;
            float r_coeff = (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.f);
            float gout = (di==0 ? dr : di==1 ? dg : db);
            trilerp += v * r_coeff * gout;
        }
        gz_grad += dwdz[corner] * (L-1) * trilerp;
    }
    
    gz_grad *= (float)(z0 != z && z1 != z);
    v_rgb[3*g_off+0] = vr + kC2G_r * gz_grad;
    v_rgb[3*g_off+1] = vg + kC2G_g * gz_grad;
    v_rgb[3*g_off+2] = vb + kC2G_b * gz_grad;
}

// Cooperative groups version for Warp-level optimization
__global__ void bilagrid_sample_backward_kernel_cooperative(
    const float* __restrict__ bilagrid,  // [N,12,L,H,W]
    const float* __restrict__ coords,  // [N,m,h,w,2]
    const float* __restrict__ rgb,  // [N,m,h,w,3]
    const float* __restrict__ v_output,  // [N,m,h,w,3]
    float* __restrict__ v_bilagrid,  // [N,12,L,H,W]
    float* __restrict__ v_rgb,  // [N,m,h,w,3]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    // Use cooperative groups for warp-level reduction
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    int wi = threadIdx.x * ((w+blockDim.x-1) / blockDim.x) + blockIdx.x;
    int hi = threadIdx.y * ((h+blockDim.y-1) / blockDim.y) + blockIdx.y;
    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    
    bool inside = (wi < w && hi < h && idx < (N*m));
    if (!inside) return;
    
    int mi = idx % m;
    int ni = idx / m;
    
    // Similar coordinate and weight calculation as hierarchical version
    // ... [coordinate calculation code] ...
    
    // Warp-level reduction for each atomicAdd operation
    for (int ci = 0; ci < 12; ++ci) {
        for (int corner = 0; corner < 8; ++corner) {
            float contribution = 0.0f; // Calculate local contribution
            
            // Warp-level sum reduction
            float warp_sum = cg::reduce(warp, contribution, cg::plus<float>());
            
            // Only one thread per warp does atomicAdd
            if (warp.thread_rank() == 0 && warp_sum != 0.0f) {
                // Calculate global address and perform single atomicAdd
                int base = ((ni*12 + ci)*L*H*W);
                atomicAdd(v_bilagrid + base + /* address calculation */, warp_sum);
            }
        }
    }
    
    // RGB gradient computation unchanged
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
    dim3 block = { 16, 16, 1 };
    dim3 bounds = {
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y,
        (N * m + block.z - 1) / block.z
    };
    
    bilagrid_sample_backward_kernel_hierarchical<<<bounds, block>>>(
        bilagrid, coords, rgb, v_output,
        v_bilagrid, v_rgb,
        N, L, H, W, m, h, w
    );
}
