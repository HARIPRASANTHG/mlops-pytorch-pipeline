"""Confirm the RTX 5070 Ti (sm_120) is actually usable by this PyTorch build."""

import torch

print(f"torch          : {torch.__version__}")
print(f"built for CUDA : {torch.version.cuda}")
print(f"cuda available : {torch.cuda.is_available()}")
print(f"arch list      : {torch.cuda.get_arch_list()}")

if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(idx)
    print(f"device         : {torch.cuda.get_device_name(idx)}")
    print(f"capability     : sm_{major}{minor}")
    if f"sm_{major}{minor}" not in torch.cuda.get_arch_list():
        raise SystemExit(
            "This wheel has no sm_120 kernels. Reinstall with:\n"
            "  pip install -r requirements/train-gpu.txt --force-reinstall"
        )
    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    print(f"matmul check   : {(x @ x).sum().item():.2f}")
    print(f"vram total     : {torch.cuda.get_device_properties(idx).total_memory / 1e9:.1f} GB")
    print("GPU is ready.")
else:
    raise SystemExit("No CUDA device visible - check the NVIDIA driver (570+ needed).")
