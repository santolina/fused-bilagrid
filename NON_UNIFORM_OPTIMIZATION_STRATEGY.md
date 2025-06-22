# Non-uniform版へのUniform最適化戦略適用可能性分析

## 概要

Uniform版で実装されている高度な最適化戦略を、non-uniform版に適用する可能性と実装戦略について詳細に分析します。

## 1. Uniform版最適化戦略の分析

### 1.1 階層リダクション戦略
```cuda
// Uniform V1での実装
__shared__ float sharedData[64];
for (int s = blockSize / 2; s > 0; s >>= 1) {
    if (tid < s)
        sharedData[tid] += sharedData[tid + s];
    __syncthreads();
}
if (tid == 0)
    atomicAdd(v_bilagrid + out_idx, sharedData[0]);
```

### 1.2 動的ワークロード分散
```cuda
// パラメータ適応調整
int mult_x = (2*w+W)/(block.x*W*target_tile_size);
int mult_y = (2*h+H)/(block.y*H*target_tile_size);
```

### 1.3 条件分岐最適化
```cuda
if (mult_x*mult_y == 1) {
    // 直接書き込み
} else if (mult_x % blockDim.x != 0) {
    // グローバルatomicAdd
} else {
    // 階層リダクション
}
```

## 2. Non-uniform版への適用可能性評価

### 2.1 階層リダクション → ✅ **高い適用可能性**

#### 現状の問題
```cuda
// Non-uniform版の現在の実装（96個のatomicAdd）
for (int ci = 0; ci < 12; ++ci) {
    atomicAdd(v_bilagrid + base + (z0*H + y0)*W + x0, f000 * grad_weight);
    atomicAdd(v_bilagrid + base + (z0*H + y0)*W + x1, f001 * grad_weight);
    // ... 8回のatomicAdd per channel
}
```

#### 適用戦略
```cuda
// 提案する改善実装
__shared__ float local_accum[12][8]; // [channels][corners]

// Phase 1: Local accumulation
for (int ci = 0; ci < 12; ++ci) {
    // Local accumulation per corner
    atomicAdd(&local_accum[ci][corner_idx], weight * grad_weight);
}
__syncthreads();

// Phase 2: Hierarchical reduction + global write
if (threadIdx.x < 8) { // 8 corners
    for (int ci = 0; ci < 12; ++ci) {
        float sum = blockReduce(local_accum[ci][threadIdx.x]);
        if (threadIdx.y == 0)
            atomicAdd(v_bilagrid + global_addr, sum);
    }
}
```

**期待効果:**
- atomicAdd回数: 96 → 96/blockSize
- メモリ競合: **80-90%削減**

### 2.2 動的ワークロード分散 → ⚠️ **条件付き適用可能**

#### 課題
- Non-uniformでは座標が不規則で事前予測困難
- サンプル密度の局所的偏りが存在

#### 適用戦略
```cuda
// 適応的ブロックサイズ調整
__device__ int estimate_local_density(float* coords, int block_start) {
    // 局所的なcoords分布を分析
    float x_var = 0, y_var = 0;
    // variance計算...
    return (x_var + y_var < threshold) ? SMALL_BLOCK : LARGE_BLOCK;
}

// 動的グリッド構成
dim3 adaptive_grid = calculate_adaptive_grid(coords_density_map);
```

**実装複雑度:** 高
**期待効果:** 中程度（10-20%改善）

### 2.3 空間的グルーピング → ✅ **非常に高い適用可能性**

#### 戦略：座標ベースクラスタリング
```cuda
// Phase 1: 座標によるソート/グルーピング
__global__ void spatial_grouping_kernel(
    float* coords, int* indices, int* group_sizes
) {
    // 空間的に近い座標をグルーピング
    // Morton encoding or Z-order curve使用
    uint32_t morton_code = morton_encode(coords[idx*2], coords[idx*2+1]);
    // Sort by morton code...
}

// Phase 2: グループ単位での処理
__global__ void grouped_backward_kernel(
    // ... grouped data
) {
    // 同一グループ内でのshared memory活用
    // 近傍アクセスの最適化
}
```

### 2.4 協調グループ → ✅ **直接適用可能**

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void non_uniform_backward_cg() {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    // Warp-level reduction
    float warp_sum = cg::reduce(warp, local_value, cg::plus<float>());
    if (warp.thread_rank() == 0) {
        atomicAdd(global_addr, warp_sum);
    }
}
```

## 3. 実装優先度と期待効果

### 3.1 Phase 1: 階層リダクション（優先度：最高）
- **実装難易度:** 中
- **期待改善:** 40-60%
- **適用対象:** Backward pass atomicAdd

```cuda
// 実装例
__shared__ float reduction_buffer[BLOCK_SIZE];
// Local accumulation → Block reduction → Single atomicAdd
```

### 3.2 Phase 2: 協調グループ（優先度：高）
- **実装難易度:** 低
- **期待改善:** 15-25%
- **適用対象:** Warp-level operations

### 3.3 Phase 3: 空間的グルーピング（優先度：中）
- **実装難易度:** 高
- **期待改善:** 20-40%
- **適用対象:** 空間局所性活用

### 3.4 Phase 4: 動的ワークロード分散（優先度：低）
- **実装難易度:** 非常に高
- **期待改善:** 10-20%
- **適用対象:** 特殊ケース最適化

## 4. 技術的制約と解決策

### 4.1 メモリ制約
```cuda
// Shared memory使用量の最適化
__shared__ float accum_buffer[MAX_CHANNELS * MAX_CORNERS]; // 動的サイズ
```

### 4.2 座標不規則性
```cuda
// 局所性検出機構
__device__ bool has_spatial_locality(float* coords, int block_size) {
    float x_range = max_x - min_x;
    float y_range = max_y - min_y;
    return (x_range < LOCALITY_THRESHOLD && y_range < LOCALITY_THRESHOLD);
}
```

### 4.3 負荷分散
```cuda
// Adaptive thread assignment
int threads_per_sample = estimate_workload(coord_complexity);
dim3 block(min(threads_per_sample, 256), 1, 1);
```

## 5. 実装ロードマップ

### Step 1: 階層リダクション実装
```cuda
// sample_backward_fast_v2.cu
__global__ void bilagrid_sample_backward_hierarchical() {
    // 1. Local accumulation in shared memory
    // 2. Block-level reduction
    // 3. Single atomicAdd per block per address
}
```

### Step 2: 協調グループ統合
```cuda
// Warp-level primitives活用
auto warp = cg::tiled_partition<32>(cg::this_thread_block());
float result = cg::reduce(warp, value, cg::plus<float>());
```

### Step 3: 空間的最適化
```cuda
// Spatial locality detection and optimization
if (detect_spatial_locality()) {
    use_shared_memory_caching();
} else {
    use_hierarchical_reduction_only();
}
```

## 6. 性能予測

### 6.1 理論的改善
- **AtomicAdd回数:** 96 → 8-12 (87-92%削減)
- **メモリ競合:** 現在の20-30%に削減
- **全体性能:** 40-70%改善

### 6.2 実装コスト
- **開発工数:** 2-3週間
- **テスト・検証:** 1週間
- **リスク:** 中程度（複雑度増加）

## 7. 結論と推奨事項

### 7.1 高い適用可能性
✅ **階層リダクション**: 直接適用、大幅改善期待  
✅ **協調グループ**: 簡単適用、中程度改善  
✅ **空間的グルーピング**: 高度実装、大幅改善  

### 7.2 実装戦略
1. **Phase 1**: 階層リダクションから開始（最大効果）
2. **Phase 2**: 協調グループで補完（実装容易）
3. **Phase 3**: 空間的最適化で仕上げ（高度最適化）

### 7.3 期待される総合効果
- **Backward pass**: 50-80%高速化
- **メモリ効率**: 大幅改善
- **スケーラビリティ**: 向上

**結論**: Uniform版の最適化戦略はnon-uniform版に**高い適用可能性**があり、特に階層リダクションは即座に実装すべき高優先度の改善です。
