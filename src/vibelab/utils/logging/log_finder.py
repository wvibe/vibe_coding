import argparse
import logging
import pathlib
import sys
from typing import Any, Dict, Optional

import yaml

# Ensure the src directory is in the Python path for imports
# This might be necessary depending on how pytest is run/configured
# Simplified path logic assuming standard structure
try:
    project_root = pathlib.Path(__file__).resolve().parents[3]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
except IndexError:
    # Handle cases where the file might be run from an unexpected location
    logging.warning(
        "Could not determine project root automatically. Assuming src is in PYTHONPATH."
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def safe_load_yaml(file_path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Safely loads a YAML file."""
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Config file not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred loading {file_path}: {e}")
        return None


def _get_wandb_config_path(wandb_run_subdir: pathlib.Path) -> Optional[pathlib.Path]:
    """Finds the config.yaml path within a WandB run directory."""
    config_path = wandb_run_subdir / "files" / "config.yaml"
    if not config_path.exists():
        # Fallback check: Config directly in the run folder root
        config_path = wandb_run_subdir / "config.yaml"
        if not config_path.exists():
            logging.debug(f"Config file not found in {wandb_run_subdir} or its 'files' subdir.")
            return None
    return config_path


def _extract_run_name_from_config(config_data: Dict[str, Any]) -> Optional[str]:
    """Extracts the Ultralytics run name from loaded WandB config data."""
    wandb_run_name_entry = config_data.get("name")
    wandb_run_name = None

    if isinstance(wandb_run_name_entry, dict):
        # Standard structure: {'desc': None, 'value': 'exp10'}
        wandb_run_name = wandb_run_name_entry.get("value")
    elif isinstance(wandb_run_name_entry, str):
        # Simpler structure possible? {'name': 'exp10'}
        wandb_run_name = wandb_run_name_entry

    return wandb_run_name


def find_wandb_run_id(ultralytics_run_dir: str, wandb_root_dir: str) -> Optional[str]:
    """
    Finds the WandB run ID corresponding to a given Ultralytics run directory.

    Args:
        ultralytics_run_dir: Path to the Ultralytics run directory
                             (e.g., 'runs/finetune/detect/exp10').
        wandb_root_dir: Path to the root WandB directory (e.g., 'wandb').

    Returns:
        The WandB run ID string if found, otherwise None.
    """
    ul_run_path = pathlib.Path(ultralytics_run_dir).resolve()
    wandb_root_path = pathlib.Path(wandb_root_dir).resolve()

    if not ul_run_path.is_dir():
        logging.error(f"Ultralytics run directory not found: {ul_run_path}")
        return None
    if not wandb_root_path.is_dir():
        logging.error(f"WandB root directory not found: {wandb_root_path}")
        return None

    ultralytics_run_name = ul_run_path.name
    logging.info(f"Searching for WandB run for Ultralytics run name: '{ultralytics_run_name}'")

    # Iterate through potential WandB run directories
    # Sort generally by timestamp; exact match depends on config content
    try:
        run_dirs = sorted(
            [d for d in wandb_root_path.iterdir() if d.is_dir() and d.name.startswith("run-")],
            key=lambda x: x.name.split("-")[1] if len(x.name.split("-")) > 1 else "",
        )
    except FileNotFoundError:  # Handle case where wandb_root_path might disappear
        logging.error(f"WandB root directory disappeared during search: {wandb_root_path}")
        return None

    for wandb_run_subdir in run_dirs:
        config_path = _get_wandb_config_path(wandb_run_subdir)
        if not config_path:
            continue  # Skip if no config found

        config_data = safe_load_yaml(config_path)
        if config_data is None:
            logging.debug(f"Skipping {wandb_run_subdir.name}: Could not load config.yaml.")
            continue

        wandb_run_name = _extract_run_name_from_config(config_data)
        if wandb_run_name is None:
            logging.debug(
                f"Skipping {wandb_run_subdir.name}: 'name' key not found or invalid format."
            )
            continue

        logging.debug(
            f"Checking WandB run {wandb_run_subdir.name}: Found name '{wandb_run_name}' in config."
        )

        # Compare with the Ultralytics run name
        if wandb_run_name == ultralytics_run_name:
            parts = wandb_run_subdir.name.split("-")
            if len(parts) >= 3:
                wandb_run_id = parts[-1]  # The last part is usually the unique ID
                logging.info(
                    f"Match found! UL run '{ultralytics_run_name}' -> WandB ID: '{wandb_run_id}'"
                )
                return wandb_run_id
            else:
                # This case should be rare with standard WandB naming
                logging.warning(
                    f"Matched name '{wandb_run_name}' in {wandb_run_subdir.name}, "
                    f"but directory name format is unexpected."
                )
                # Optionally return None or continue searching? Let's continue for now.

    logging.warning(
        f"No matching WandB run found for Ultralytics run name: '{ultralytics_run_name}'"
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the WandB run ID for a given Ultralytics run directory."
    )
    parser.add_argument(
        "ultralytics_run_dir",
        type=str,
        help="Path to the Ultralytics run directory (e.g., runs/finetune/detect/exp10)",
    )
    parser.add_argument(
        "wandb_root_dir",
        type=str,
        nargs="?",  # Make wandb_root_dir optional
        default="wandb",  # Default to 'wandb' in the current directory
        help="Path to the root WandB directory (default: ./wandb)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose debug logging."
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    wandb_id = find_wandb_run_id(args.ultralytics_run_dir, args.wandb_root_dir)

    if wandb_id:
        print(f"WandB Run ID: {wandb_id}")
    else:
        print("Matching WandB run ID not found.")
        exit(1)  # Exit with error code if not found
