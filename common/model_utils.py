"""Small model/GPU utilities shared across pipelines: parameter counting, FLOPs
reporting, and GPU memory cleanup."""

import gc

import torch


def count_parameters(model):
    """Return ``(total_params, trainable_params)``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_stats(model, input_size=(1, 3, 1024, 1024), device="cpu"):
    """Print trainable parameter count and (if ``thop`` is installed) FLOPs."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total / 1e6:.2f}M")
    try:
        from thop import profile
        dummy = torch.randn(*input_size).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")
    except ImportError:
        print("Install thop (`pip install thop`) for FLOPs calculation.")


def clean_gpu():
    """Free cached GPU memory on every visible device."""
    gc.collect()
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    n = torch.cuda.device_count()
    for gpu_id in range(n):
        with torch.cuda.device(gpu_id):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    print(f"GPU memory cleaned on {n} GPU(s).")
