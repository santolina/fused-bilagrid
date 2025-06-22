#!/usr/bin/env python3
"""
高速化されたfused bilagridの性能テストとベンチマーク
"""

import time
import torch
import numpy as np
from typing import Tuple, Dict, Any

try:
    import fused_bilagrid
    HAS_FUSED_BILAGRID = True
except ImportError:
    HAS_FUSED_BILAGRID = False
    print("Warning: fused_bilagrid not available, using mock implementation")

def generate_test_data(N: int = 1, L: int = 8, H: int = 16, W: int = 16, 
                      m: int = 4, h: int = 32, w: int = 32) -> Tuple[torch.Tensor, ...]:
    """テストデータを生成"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # BilaGrid: [N, 12, L, H, W]
    bilagrid = torch.randn(N, 12, L, H, W, device=device, dtype=torch.float32)
    
    # Coordinates: [N, m, h, w, 2] - normalized coordinates [0, 1]
    coords = torch.rand(N, m, h, w, 2, device=device, dtype=torch.float32)
    
    # RGB: [N, m, h, w, 3]
    rgb = torch.rand(N, m, h, w, 3, device=device, dtype=torch.float32)
    
    # Output gradients for backward pass
    v_output = torch.randn(N, m, h, w, 3, device=device, dtype=torch.float32)
    
    return bilagrid, coords, rgb, v_output

def benchmark_function(func, *args, iterations: int = 100, warmup: int = 10) -> Dict[str, float]:
    """関数のベンチマークを実行"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmark")
        return {"mean_time": 0.0, "std_time": 0.0, "throughput": 0.0}
    
    # Warmup
    for _ in range(warmup):
        func(*args)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times)
    return {
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "throughput": 1.0 / np.mean(times)
    }

def test_forward_implementations():
    """Forward実装のテストとベンチマーク"""
    print("=" * 60)
    print("FORWARD KERNEL BENCHMARKS")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {"N": 1, "L": 8, "H": 16, "W": 16, "m": 4, "h": 32, "w": 32, "name": "Small"},
        {"N": 2, "L": 16, "H": 32, "W": 32, "m": 8, "h": 64, "w": 64, "name": "Medium"},
        {"N": 4, "L": 32, "H": 64, "W": 64, "m": 16, "h": 128, "w": 128, "name": "Large"},
    ]
    
    for config in configs:
        config_name = config.pop("name")
        print(f"\n{config_name} Configuration: {config}")
        
        # Generate test data
        bilagrid, coords, rgb, _ = generate_test_data(**config)
        
        if not HAS_FUSED_BILAGRID:
            print("  Skipping benchmark - fused_bilagrid not available")
            continue
            
        # Test original implementation
        def original_forward():
            return fused_bilagrid.sample_forward(bilagrid, coords, rgb)
        
        # Test fast implementation (if available)
        def fast_forward():
            return fused_bilagrid.sample_forward_fast(bilagrid, coords, rgb)
        
        try:
            print("  Testing original implementation...")
            original_stats = benchmark_function(original_forward, iterations=50)
            print(f"    Original: {original_stats['mean_time']:.4f}s ± {original_stats['std_time']:.4f}s")
            
            print("  Testing fast implementation...")
            fast_stats = benchmark_function(fast_forward, iterations=50)
            print(f"    Fast: {fast_stats['mean_time']:.4f}s ± {fast_stats['std_time']:.4f}s")
            
            speedup = original_stats['mean_time'] / fast_stats['mean_time']
            print(f"    Speedup: {speedup:.2f}x")
            
            # Verify correctness
            output_orig = original_forward()
            output_fast = fast_forward()
            max_diff = torch.max(torch.abs(output_orig - output_fast)).item()
            print(f"    Max difference: {max_diff:.2e}")
            
        except Exception as e:
            print(f"    Error: {e}")

def test_backward_implementations():
    """Backward実装のテストとベンチマーク"""
    print("\n" + "=" * 60)
    print("BACKWARD KERNEL BENCHMARKS")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {"N": 1, "L": 8, "H": 16, "W": 16, "m": 4, "h": 32, "w": 32, "name": "Small"},
        {"N": 2, "L": 16, "H": 32, "W": 32, "m": 8, "h": 64, "w": 64, "name": "Medium"},
        {"N": 4, "L": 32, "H": 64, "W": 64, "m": 16, "h": 128, "w": 128, "name": "Large"},
    ]
    
    for config in configs:
        config_name = config.pop("name")
        print(f"\n{config_name} Configuration: {config}")
        
        # Generate test data
        bilagrid, coords, rgb, v_output = generate_test_data(**config)
        
        if not HAS_FUSED_BILAGRID:
            print("  Skipping benchmark - fused_bilagrid not available")
            continue
        
        # Initialize gradient tensors
        v_bilagrid = torch.zeros_like(bilagrid)
        v_coords = torch.zeros_like(coords)
        v_rgb = torch.zeros_like(rgb)
        
        v_bilagrid_fast = torch.zeros_like(bilagrid)
        v_coords_fast = torch.zeros_like(coords)
        v_rgb_fast = torch.zeros_like(rgb)
        
        # Test original implementation
        def original_backward():
            v_bilagrid.zero_()
            v_coords.zero_()
            v_rgb.zero_()
            return fused_bilagrid.sample_backward(
                bilagrid, coords, rgb, v_output, v_bilagrid, v_coords, v_rgb)
        
        # Test fast implementation
        def fast_backward():
            v_bilagrid_fast.zero_()
            v_coords_fast.zero_()
            v_rgb_fast.zero_()
            return fused_bilagrid.sample_backward_fast(
                bilagrid, coords, rgb, v_output, v_bilagrid_fast, v_coords_fast, v_rgb_fast)
        
        try:
            print("  Testing original implementation...")
            original_stats = benchmark_function(original_backward, iterations=50)
            print(f"    Original: {original_stats['mean_time']:.4f}s ± {original_stats['std_time']:.4f}s")
            
            print("  Testing fast implementation...")
            fast_stats = benchmark_function(fast_backward, iterations=50)
            print(f"    Fast: {fast_stats['mean_time']:.4f}s ± {fast_stats['std_time']:.4f}s")
            
            speedup = original_stats['mean_time'] / fast_stats['mean_time']
            print(f"    Speedup: {speedup:.2f}x")
            
            # Verify correctness
            original_backward()
            fast_backward()
            
            max_diff_bilagrid = torch.max(torch.abs(v_bilagrid - v_bilagrid_fast)).item()
            max_diff_coords = torch.max(torch.abs(v_coords - v_coords_fast)).item()
            max_diff_rgb = torch.max(torch.abs(v_rgb - v_rgb_fast)).item()
            
            print(f"    Max difference (bilagrid): {max_diff_bilagrid:.2e}")
            print(f"    Max difference (coords): {max_diff_coords:.2e}")
            print(f"    Max difference (rgb): {max_diff_rgb:.2e}")
            
        except Exception as e:
            print(f"    Error: {e}")

def profile_memory_usage():
    """メモリ使用量のプロファイリング"""
    print("\n" + "=" * 60)
    print("MEMORY USAGE PROFILING")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling")
        return
    
    # Generate large test data
    bilagrid, coords, rgb, v_output = generate_test_data(
        N=4, L=32, H=64, W=64, m=16, h=128, w=128)
    
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    print(f"Initial GPU memory: {initial_memory / 1024**2:.2f} MB")
    
    if HAS_FUSED_BILAGRID:
        # Test forward pass
        try:
            output = fused_bilagrid.sample_forward(bilagrid, coords, rgb)
            forward_memory = torch.cuda.memory_allocated()
            print(f"After forward pass: {forward_memory / 1024**2:.2f} MB")
            print(f"Forward memory increase: {(forward_memory - initial_memory) / 1024**2:.2f} MB")
            
            # Test backward pass
            v_bilagrid = torch.zeros_like(bilagrid)
            v_coords = torch.zeros_like(coords)
            v_rgb = torch.zeros_like(rgb)
            
            fused_bilagrid.sample_backward(
                bilagrid, coords, rgb, v_output, v_bilagrid, v_coords, v_rgb)
            backward_memory = torch.cuda.memory_allocated()
            print(f"After backward pass: {backward_memory / 1024**2:.2f} MB")
            print(f"Backward memory increase: {(backward_memory - forward_memory) / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"Error during memory profiling: {e}")

def main():
    """メイン実行関数"""
    print("Fused BilaGrid Optimization Benchmark")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    test_forward_implementations()
    test_backward_implementations()
    profile_memory_usage()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
