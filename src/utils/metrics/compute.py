import logging
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def get_model_params(model):
    """Attempts to retrieve the total number of parameters for a given model.

    Primarily designed for Ultralytics YOLO models, attempting to access
    pre-calculated parameter counts. Falls back to summing PyTorch tensor parameters.

    Args:
        model: The loaded model object (e.g., ultralytics.YOLO model instance).

    Returns:
        int or None: The total number of parameters, or None if unable to determine.
    """
    try:
        # Ultralytics models often store param count in model.model.info()
        # The last element of the tuple returned by info(detailed=False) is usually params
        # Correction: The second element (index 1) seems to be parameters based on common ultralytics output
        # Accessing model.model assumes the YOLO object structure
        if hasattr(model, 'model') and hasattr(model.model, 'info'):
            model_info_tuple = model.model.info(detailed=False)
            if len(model_info_tuple) > 1:
                params = model_info_tuple[1] # Assuming index 1 is parameters
            else:
                params = None # Handle unexpected tuple length
            if isinstance(params, (int, float)) and params > 0:
                log.debug("Retrieved params from model.model.info()")
                return int(params)
            else:
                log.warning("Could not parse params from model.model.info(), attempting fallback.")
        else:
            log.debug("model.model.info() not available, attempting fallback.")

        # Fallback: Sum parameters directly from PyTorch model
        # Ensure we have a torch.nn.Module
        pytorch_model = model.model if hasattr(model, 'model') else model
        if isinstance(pytorch_model, torch.nn.Module):
            total_params = sum(p.numel() for p in pytorch_model.parameters())
            # Often people report trainable parameters
            trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
            log.debug(f"Calculated params via torch: Total={total_params}, Trainable={trainable_params}")
            # Return trainable by default as it's more common for reporting
            if trainable_params > 0:
                return trainable_params
            elif total_params > 0:
                return total_params # Return total if no trainable found
            else:
                 log.warning("Could not count parameters using torch fallback.")
                 return None
        else:
            log.warning("Model object is not a recognized torch.nn.Module for fallback.")
            return None

    except Exception as e:
        log.error(f"Error retrieving model parameters: {e}", exc_info=True)
        return None

def get_peak_gpu_memory_mb(device=None):
    """Gets the peak GPU memory allocated on a specific device since the last reset.

    Args:
        device (str or torch.device, optional): The CUDA device to query
            (e.g., 'cuda:0', torch.device('cuda:1')). Defaults to the current
            default CUDA device if None.

    Returns:
        float or None: Peak memory allocated in MiB, or None if CUDA is unavailable
                       or the device is not a CUDA device.
    """
    if not torch.cuda.is_available():
        log.info("CUDA not available, cannot report GPU memory.")
        return None

    try:
        if device is None:
            # Get default CUDA device index
            device_idx = torch.cuda.current_device()
        elif isinstance(device, str):
            if not device.startswith('cuda'):
                 log.info(f"Device '{device}' is not a CUDA device.")
                 return None
            # Extract index if present, default to 0 otherwise
            try:
                device_idx = int(device.split(':')[-1]) if ':' in device else 0
            except ValueError:
                log.error(f"Invalid CUDA device string: {device}")
                return None
        elif isinstance(device, torch.device):
            if device.type != 'cuda':
                log.info(f"Device '{device}' is not a CUDA device.")
                return None
            device_idx = device.index if device.index is not None else torch.cuda.current_device()
        elif isinstance(device, int): # Add check for integer device ID
            if device < 0 or device >= torch.cuda.device_count():
                log.error(f"CUDA device index {device} out of range.")
                return None
            device_idx = device # Use the integer directly
        else:
             log.error(f"Invalid device type: {type(device)}")
             return None

        # No need for second check, handled above
        # if device_idx >= torch.cuda.device_count():
        #      log.error(f"CUDA device index {device_idx} out of range.")
        #      return None

        # Get peak memory in bytes and convert to MiB (1024*1024)
        peak_mem_bytes = torch.cuda.max_memory_allocated(device=device_idx)
        peak_mem_mb = peak_mem_bytes / (1024 * 1024)
        log.debug(f"Peak memory on device {device_idx}: {peak_mem_mb:.2f} MiB")
        return peak_mem_mb

    except Exception as e:
        log.error(f"Error retrieving peak GPU memory: {e}", exc_info=True)
        return None

# Remember to call torch.cuda.reset_peak_memory_stats(device) before the operation
# you want to measure.