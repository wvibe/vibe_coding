# Logging Utilities Design Notes

This document outlines the design choices and rationale behind the functions in the `src/utils/logging/` directory, which contain utilities for logging, tracking, and monitoring experiments.

---

## `log_finder.py` - Experiment Log ID Finder

This module provides functionality for finding and matching experiment run IDs between local training directories and remote logging services like Weights & Biases (W&B).

### Key Functions

#### `find_wandb_run_id(ultralytics_run_dir, wandb_root_dir)`

- **Purpose:** Maps local Ultralytics run directories to their corresponding W&B run IDs.
- **Input:**
  - `ultralytics_run_dir`: Path to an Ultralytics run directory (e.g., 'runs/finetune/detect/exp10')
  - `wandb_root_dir`: Path to the W&B root directory (typically 'wandb' in the project root)
- **Logic:**
  1. Extracts the run name from the Ultralytics directory path
  2. Searches through W&B run directories for a matching run name in their config files
  3. When a match is found, extracts the W&B run ID from the directory name
- **Output:** The W&B run ID as a string if found, otherwise None.

### Helper Functions

#### `safe_load_yaml(file_path)`

- **Purpose:** Safely loads a YAML file with comprehensive error handling.
- **Logic:** Attempts to load the YAML file, handling FileNotFound, YAML parsing errors, and other exceptions.
- **Output:** The parsed YAML content as a dict if successful, otherwise None.

#### `_get_wandb_config_path(wandb_run_subdir)`

- **Purpose:** Locates the config.yaml file within a W&B run directory.
- **Logic:** Checks both the 'files' subdirectory and the root of the run directory.
- **Output:** The path to the config file if found, otherwise None.

#### `_extract_run_name_from_config(config_data)`

- **Purpose:** Extracts the Ultralytics run name from loaded W&B config data.
- **Logic:** Handles different possible structures of the config data for the run name.
- **Output:** The run name as a string if found, otherwise None.

### Command-Line Interface

The module includes a command-line interface for direct use as a script:

```bash
python -m src.utils.logging.log_finder path/to/ultralytics/run [path/to/wandb/dir] [-v]
```

- **Arguments:**
  - Required: `ultralytics_run_dir` (path to the Ultralytics run directory)
  - Optional: `wandb_root_dir` (path to the W&B directory, defaults to './wandb')
  - Optional: `-v` or `--verbose` flag to enable debug logging

- **Output:** Prints the matching W&B run ID to stdout if found, otherwise exits with code 1.