#!/usr/bin/env python3
"""
Script to diagnose and help mitigate TaskAlignedAssigner CUDA OOM issues.
This script provides utilities to:
1. Monitor GPU memory usage
2. Estimate memory requirements for TaskAlignedAssigner
3. Suggest optimal batch sizes
4. Apply memory optimization patches if needed
"""

import torch
import math
import os
import sys
from pathlib import Path

def get_gpu_memory_info():
    """Get GPU memory information for all available GPUs"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return None

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(i) / (1024**3)    # GB
        total = props.total_memory / (1024**3)                  # GB
        free = total - allocated

        gpu_info.append({
            'device': i,
            'name': props.name,
            'total_gb': total,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'utilization': (allocated / total) * 100
        })

    return gpu_info

def estimate_tal_memory_usage(batch_size, img_size=640, max_objects=50, num_anchors=None):
    """
    Estimate memory usage for TaskAlignedAssigner operations

    Args:
        batch_size: Training batch size
        img_size: Image size (assumes square)
        max_objects: Maximum number of objects per image
        num_anchors: Number of anchors (if None, calculated from img_size)
    """
    if num_anchors is None:
        # Approximate anchor calculation for YOLOv11
        # Typically has 3 detection heads with different strides
        strides = [8, 16, 32]
        num_anchors = sum((img_size // stride) ** 2 for stride in strides)

    # Key memory-intensive operations in TaskAlignedAssigner:
    # 1. overlaps tensor: (batch_size, max_objects, num_anchors)
    # 2. bbox_scores tensor: (batch_size, max_objects, num_anchors)
    # 3. Expanded tensors for IoU calculation

    # Basic tensors (float32 = 4 bytes)
    overlaps_memory = batch_size * max_objects * num_anchors * 4
    bbox_scores_memory = batch_size * max_objects * num_anchors * 4

    # Expanded tensors during IoU calculation (temporary but large)
    # pd_boxes expanded: (batch_size, max_objects, num_anchors, 4)
    # gt_boxes expanded: (batch_size, max_objects, num_anchors, 4)
    expanded_memory = batch_size * max_objects * num_anchors * 4 * 2 * 4  # 2 tensors, 4 coords each

    total_memory_mb = (overlaps_memory + bbox_scores_memory + expanded_memory) / (1024**2)

    return {
        'batch_size': batch_size,
        'img_size': img_size,
        'max_objects': max_objects,
        'num_anchors': num_anchors,
        'overlaps_mb': overlaps_memory / (1024**2),
        'bbox_scores_mb': bbox_scores_memory / (1024**2),
        'expanded_mb': expanded_memory / (1024**2),
        'total_mb': total_memory_mb,
        'total_gb': total_memory_mb / 1024
    }

def suggest_optimal_batch_size(target_memory_gb=18, img_size=640, max_objects=50):
    """Suggest optimal batch size to avoid TaskAlignedAssigner OOM"""
    target_memory_mb = target_memory_gb * 1024

    # Binary search for optimal batch size
    low, high = 1, 256
    best_batch_size = 1

    while low <= high:
        mid = (low + high) // 2
        est = estimate_tal_memory_usage(mid, img_size, max_objects)

        if est['total_mb'] <= target_memory_mb:
            best_batch_size = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_batch_size

def apply_memory_optimization_env_vars():
    """Apply PyTorch memory optimization environment variables"""
    optimizations = {
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:128',
        'CUDA_LAUNCH_BLOCKING': '0',  # Don't block for debugging (faster)
    }

    print("Applying memory optimization environment variables:")
    for key, value in optimizations.items():
        os.environ[key] = value
        print(f"  {key} = {value}")

    return optimizations

def main():
    print("=== TaskAlignedAssigner Memory Diagnostic Tool ===\n")

    # Check GPU memory
    print("1. GPU Memory Status:")
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        for gpu in gpu_info:
            print(f"  GPU {gpu['device']} ({gpu['name']}): "
                  f"{gpu['free_gb']:.1f}GB free / {gpu['total_gb']:.1f}GB total "
                  f"({gpu['utilization']:.1f}% used)")
    print()

    # Estimate memory for different batch sizes
    print("2. TaskAlignedAssigner Memory Estimates:")
    batch_sizes = [32, 64, 96, 128, 192, 256]

    for batch_size in batch_sizes:
        est = estimate_tal_memory_usage(batch_size)
        print(f"  Batch {batch_size:3d}: {est['total_gb']:.2f}GB "
              f"(overlaps: {est['overlaps_mb']:.0f}MB, "
              f"expanded: {est['expanded_mb']:.0f}MB)")
    print()

    # Suggest optimal batch size
    if gpu_info:
        # Use 80% of the smallest GPU's free memory as target
        min_free_memory = min(gpu['free_gb'] for gpu in gpu_info)
        target_memory = min_free_memory * 0.8

        optimal_batch = suggest_optimal_batch_size(target_memory)
        print(f"3. Suggested batch size: {optimal_batch} "
              f"(targeting {target_memory:.1f}GB for TaskAlignedAssigner)")
        print()

    # Apply optimizations
    print("4. Applying PyTorch memory optimizations:")
    apply_memory_optimization_env_vars()
    print()

    print("5. Additional Recommendations:")
    print("  - Use val=True with max_det=100 in training config")
    print("  - Reduce batch size further if needed (64, 48, 32)")
    print("  - Consider reducing workers count")
    print("  - Monitor nvidia-smi during training")
    print("  - Set PYTORCH_CUDA_ALLOC_CONF environment variable")
    print("  - If issue persists, try reducing image size from 640 to 512")

if __name__ == "__main__":
    main()