# GPU環境での検証手順書

## 前提条件
- NVIDIA GPU搭載環境 (Compute Capability 7.5以上推奨)
- CUDA 11.0以上
- Python 3.8以上
- PyTorch (CUDA版)

## 環境別セットアップ手順

### Google Colab Pro/Pro+

```python
# GPU確認
!nvidia-smi
!nvcc --version

# リポジトリクローン
!git clone https://github.com/your-username/fused-bilagrid.git
%cd fused-bilagrid  
!git checkout dev_fast

# 依存関係
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ビルド
!python setup.py build_ext --inplace

# 基本テスト
!python -c "import fused_bilagrid_cuda as _C; print('Available functions:', dir(_C))"
```

### AWS EC2 (p3.2xlarge, g4dn.xlarge等)

```bash
# Deep Learning AMI (Ubuntu 20.04) 推奨
# インスタンス起動後

# リポジトリセットアップ
git clone https://github.com/your-username/fused-bilagrid.git
cd fused-bilagrid
git checkout dev_fast

# 環境確認
nvidia-smi
nvcc --version
python --version

# PyTorch確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# ビルド
python setup.py build_ext --inplace

# テスト実行
python tests/benchmark_fast.py
```

### Azure ML Compute Instance

```bash
# Compute Instance (GPU) 作成後
# Terminal で実行

git clone <repository>
cd fused-bilagrid
git checkout dev_fast

# 環境準備
conda activate azureml_py38_PT_TF

# ビルド・テスト
python setup.py build_ext --inplace
python tests/test_bilagrid.py
```

### ローカル環境 (WSL2 + CUDA)

```bash
# WSL2でのCUDA設定後
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# プロジェクトセットアップ
git clone <repository>
cd fused-bilagrid
git checkout dev_fast

# ビルド
python setup.py build_ext --inplace
```

## 検証項目

### 1. ビルド確認
```bash
python setup.py build_ext --inplace
echo "Build Status: $?"
```

### 2. モジュールインポート
```python
import fused_bilagrid_cuda as _C

# 利用可能な関数確認
available_functions = dir(_C)
print("Available functions:")
for func in available_functions:
    if not func.startswith('_'):
        print(f"  - {func}")

# 新しいカーネル確認
required_functions = [
    'bilagrid_sample_forward_adaptive',
    'bilagrid_sample_backward_hierarchical'
]

for func in required_functions:
    if hasattr(_C, func):
        print(f"✓ {func} - Available")
    else:
        print(f"✗ {func} - Missing")
```

### 3. 基本動作テスト
```python
import torch
import fused_bilagrid_cuda as _C

# テストデータ生成
device = 'cuda'
N, L, H, W = 1, 8, 16, 16
m, h, w = 4, 32, 32

bilagrid = torch.randn(N, 12, L, H, W, device=device)
coords = torch.rand(N, m, h, w, 2, device=device)
rgb = torch.rand(N, m, h, w, 3, device=device)

# 各カーネルのテスト
try:
    # Original
    output_orig = _C.bilagrid_sample_forward(bilagrid, coords, rgb)
    print("✓ Original forward - OK")
except Exception as e:
    print(f"✗ Original forward - Error: {e}")

try:
    # Adaptive
    output_adaptive = _C.bilagrid_sample_forward_adaptive(bilagrid, coords, rgb)
    print("✓ Adaptive forward - OK")
except Exception as e:
    print(f"✗ Adaptive forward - Error: {e}")

# 結果比較
if 'output_orig' in locals() and 'output_adaptive' in locals():
    diff = torch.abs(output_orig - output_adaptive).max().item()
    print(f"Max difference: {diff}")
    if diff < 1e-5:
        print("✓ Outputs match (within tolerance)")
    else:
        print("⚠ Outputs differ significantly")
```

### 4. 性能ベンチマーク
```python
# benchmark_fast.py実行
python tests/benchmark_fast.py

# 期待される出力例:
# Testing kernel availability:
#   bilagrid_sample_forward: True
#   bilagrid_sample_forward_adaptive: True
#   bilagrid_sample_backward_hierarchical: True
#
# Forward Pass Comparison:
# ==================================================
# Original    : 2.1234ms ± 0.0456ms  
# Adaptive    : 1.5678ms ± 0.0234ms  (1.36x speedup)
#
# Backward Pass Comparison: 
# ==================================================
# Original    : 3.4567ms ± 0.0789ms
# Hierarchical: 2.1234ms ± 0.0345ms  (1.63x speedup)
```

## トラブルシューティング

### ビルドエラー
```bash
# CUDA_HOME設定
export CUDA_HOME=/usr/local/cuda

# コンパイラ確認
which nvcc
nvcc --version

# PyTorch CUDA版確認
python -c "import torch; print(torch.version.cuda)"
```

### メモリエラー
```python
# GPU メモリ確認
torch.cuda.empty_cache()
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
```

### 性能が改善されない場合
- GPU Compute Capability確認 (7.5以上推奨)
- データサイズ確認 (小さすぎるとオーバーヘッドが支配的)
- 空間局所性確認 (ランダムデータでは効果限定的)

## 期待される性能改善

| データ特性 | Forward改善 | Backward改善 | 条件 |
|-----------|------------|-------------|------|
| 高空間局所性 | 30-50% | 40-60% | 座標分散 < 0.01 |
| 中空間局所性 | 10-25% | 20-35% | 座標分散 0.01-0.1 |
| 低空間局所性 | 5-15% | 10-20% | 座標分散 > 0.1 |
