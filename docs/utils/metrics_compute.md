# Computational Metrics Utilities Design Notes

This document outlines the design choices and rationale behind the functions in `src/utils/metrics/compute.py`.

## `get_model_params(model)`

### Purpose

Retrieves the total number of parameters (typically trainable) for a given model object. This is a common metric for reporting model size.

### Logic

1.  **Input:** Takes a model object, which is expected to be either an `ultralytics.YOLO` instance or a `torch.nn.Module`.
2.  **Ultralytics `.info()` Attempt:** It first checks if the model is likely an Ultralytics YOLO object (`hasattr(model, 'model')`) and if its internal model has an `.info()` method. If so, it calls `model.model.info(detailed=False)` and attempts to parse the parameter count (often the last element of the returned tuple).
3.  **PyTorch Fallback:** If the `.info()` method is unavailable or returns non-numeric parameters, it falls back to treating the model (or `model.model`) as a standard `torch.nn.Module`. It calculates the sum of elements (`numel()`) for all parameters, prioritizing the count of trainable parameters (`p.requires_grad`).
4.  **Output:** Returns the parameter count as an integer, prioritizing the `.info()` result, then trainable parameters, then total parameters. Returns `None` if the parameter count cannot be determined.

### Testing

Unit tests use mock objects to simulate both the Ultralytics `.info()` structure and a basic `torch.nn.Module` to verify the different retrieval paths and the fallback logic. Testing with real, complex models is deferred to integration testing within the evaluation script.

## `get_peak_gpu_memory_mb(device=None)`

### Purpose

Reports the peak GPU memory usage (in MiB) on a specified CUDA device since the last time the memory statistics were reset for that device.

### Logic

1.  **Input:** Takes an optional `device` specification (string like `'cuda:0'`, `torch.device` object, or `None` for default CUDA device).
2.  **CUDA Check:** Immediately returns `None` if `torch.cuda.is_available()` is `False`.
3.  **Device Parsing:** Determines the target CUDA device index based on the input `device`.
4.  **Validation:** Checks if the specified device is a CUDA device and if the index is valid.
5.  **Memory Retrieval:** If valid, calls `torch.cuda.max_memory_allocated(device=device_idx)` to get the peak memory in bytes.
6.  **Conversion:** Converts the byte value to MiB (dividing by 1024*1024).
7.  **Output:** Returns the peak memory in MiB as a float, or `None` if CUDA is unavailable, the device is not CUDA, or an error occurs.

### Important Usage Note

This function only *reads* the peak value. To measure the peak memory for a specific operation (like a prediction loop), you **must** call `torch.cuda.reset_peak_memory_stats(device)` *before* starting the operation.

### Testing

Directly unit testing CUDA memory usage is unreliable and environment-dependent. The provided unit tests use `unittest.mock.patch` to:
*   Simulate scenarios where CUDA is available or unavailable.
*   Mock the return value of `torch.cuda.max_memory_allocated` to verify the conversion logic.
*   Test the handling of non-CUDA devices and invalid device indices.
Actual memory usage verification requires running the evaluation script on a target machine with a CUDA-enabled GPU.