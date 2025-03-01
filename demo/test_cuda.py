import sys
import torch
import subprocess
from pathlib import Path
import os

def check_cuda_environment():
    """全面检查CUDA环境"""
    print("=== 系统信息 ===")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    print("\n=== CUDA信息 ===")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU索引: {torch.cuda.current_device()}")
        
        # GPU内存信息
        print(f"\nGPU内存信息:")
        print(f"总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"当前分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"当前缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # 简单的GPU计算测试
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            print("\nGPU矩阵计算测试: 成功")
        except Exception as e:
            print(f"\nGPU计算测试失败: {str(e)}")
    
    print("\n=== NVIDIA驱动信息 ===")
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
        print(nvidia_smi.decode())
    except:
        print("无法获取NVIDIA驱动信息")
    
    print("\n=== CUDA路径检查 ===")
    cuda_paths = [
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),  # Windows
        Path("/usr/local/cuda"),  # Linux
        Path(os.environ.get("CUDA_PATH", ""))  # 环境变量
    ]
    
    for path in cuda_paths:
        if path.exists():
            print(f"找到CUDA安装: {path}")
            nvcc_path = path / "bin" / "nvcc"
            if nvcc_path.exists() or (nvcc_path.with_suffix(".exe")).exists():
                print(f"找到NVCC: {nvcc_path}")

if __name__ == "__main__":
    check_cuda_environment()
