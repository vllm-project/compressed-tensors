import json
import os
import shutil
from pathlib import Path

from compressed_tensors.entrypoints.convert.file_utils import (
    find_safetensors_index_path,
    get_checkpoint_files,
)
from compressed_tensors.entrypoints.convert.save_utils import (
    update_safetensors_index,
)
from loguru import logger
from safetensors.torch import load_file, save_file

__all__ = ["consolidate_checkpoint_tensors"]


def get_module_name(tensor_name: str) -> str:
    """
    Extract module name from tensor name.
    Module name is everything before the last component (e.g., "weight", "weight_scale_inv")

    :param tensor_name: full tensor name like "model.layers.60.mlp.experts.84.up_proj.weight"
    :return: module name like "model.layers.60.mlp.experts.84.up_proj"
    """
    parts = tensor_name.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else tensor_name


def consolidate_checkpoint_tensors(
    model_stub: str | os.PathLike,
    save_directory: str | os.PathLike,
):
    """
    Consolidate tensors from the same module into a single safetensors file. This
    should be idempotent.

<<<<<<< Updated upstream
    This function processes every safetensor file in the checkpoint in sorted order
    and consolidates tensors for each module. It assumes that if a module's tensors
    are split, they will be in a file and the immediately next file. For each file,
    it checks if any of its module's tensors appear in the next file, and if so, moves
    them back to the current file.
=======
    This function processes safetensors files in sorted order and consolidates tensors
    for each module. For each file, it scans all subsequent files to find any tensors
    belonging to the same modules, and moves them into the current file. This ensures
    that all tensors for a given module (e.g., "model.layers.60.mlp.experts.84.up_proj")
    are stored together in a single safetensors file.
>>>>>>> Stashed changes

    :param model_stub: huggingface model hub ID or path to local directory containing
        the input safetensors files
    :param save_directory: directory to save consolidated files to
    """

    # Determine if in-place based on resolved paths
    if os.path.exists(model_stub):
        in_place = str(model_stub) == str(save_directory)
    else:
        in_place = False

    # Get all checkpoint files (handles both local paths and HF hub)
    model_files = get_checkpoint_files(model_stub)

    save_path = Path(save_directory)

    # Create save directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)

    # Filter to safetensors files (excluding index) and sort by filename
    safetensors_files = sorted(
        [
            (file_path, resolved_path)
            for file_path, resolved_path in model_files.items()
            if file_path.endswith(".safetensors")
            and not file_path.endswith(".index.json")
        ],
        key=lambda x: x[0],  # Sort by relative path
    )

    if not safetensors_files:
        logger.info("No safetensors files found")
        return

    logger.info(f"Found {len(safetensors_files)} safetensors files to process")

    # Track which files will be written and which should be skipped
    files_to_skip = set()  # filenames that will be empty/removed
    weight_map = {}

    # Create a mapping of filename to (file_path, resolved_path) for easy lookup
    file_paths_map = {
        os.path.basename(file_path): (file_path, resolved_path)
        for file_path, resolved_path in safetensors_files
    }
    filenames = sorted(file_paths_map.keys())

    # Process each file and consolidate its modules from all subsequent files
    for i, current_filename in enumerate(filenames):
        # Skip if file was marked for removal in a previous iteration
        if current_filename in files_to_skip:
            continue

        # Load current file
        _, current_resolved_path = file_paths_map[current_filename]
        current_tensors = load_file(current_resolved_path)

        # Get all modules in current file
        current_modules = {get_module_name(name) for name in current_tensors.keys()}

        # Check all subsequent files for matching module tensors
        for j in range(i + 1, len(filenames)):
            other_filename = filenames[j]

            # Skip if already marked for removal
            if other_filename in files_to_skip:
                continue

            # Load other file
            _, other_resolved_path = file_paths_map[other_filename]
            other_tensors = load_file(other_resolved_path)

            # Find tensors in other file that belong to modules in current file
            tensors_to_move = {}
            for tensor_name in list(other_tensors.keys()):
                module_name = get_module_name(tensor_name)
                if module_name in current_modules:
                    tensors_to_move[tensor_name] = other_tensors[tensor_name]
                    del other_tensors[tensor_name]

            # Move tensors to current file and save/remove other file
            if tensors_to_move:
                logger.info(
                    f"Moving {len(tensors_to_move)} tensors from {other_filename} to {current_filename}"
                )
                current_tensors.update(tensors_to_move)

                # Save or mark other file for removal
                if len(other_tensors) == 0:
                    files_to_skip.add(other_filename)
                    logger.info(f"Marking {other_filename} for removal (now empty)")
                else:
                    # Save the modified other file immediately
                    other_output_file = save_path / other_filename
                    save_file(other_tensors, other_output_file)
                    logger.info(f"Saved {other_filename} with {len(other_tensors)} tensors")

        # Save current file
        current_output_file = save_path / current_filename
        save_file(current_tensors, current_output_file)
        logger.info(f"Saved {current_filename} with {len(current_tensors)} tensors")

        # Update weight map
        for tensor_name in current_tensors.keys():
            weight_map[tensor_name] = current_filename

    # Delete empty safetensors files when operating in-place
    if in_place:
        for filename in files_to_skip:
            file_path = save_path / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Removed empty file {filename}")

    # Copy all non-safetensors files to save directory (if not in-place)
    if not in_place:
        for file_path, resolved_path in model_files.items():
            # Skip safetensors files (already handled) and the index (will be updated)
            if file_path.endswith(".safetensors"):
                continue

            dest_path = save_path / file_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved_path, dest_path)
            logger.info(f"Copied {file_path} to save directory")

    # Update safetensors index
    # For HF hub models, check if index exists in model_files
    has_index = any(fp.endswith("safetensors.index.json") for fp in model_files.keys())
    if has_index or len(safetensors_files) > 1:
        # Calculate total size from saved files
        total_size = 0
        for file_name in set(weight_map.values()):
            file_path = save_path / file_name
            if file_path.exists():
                # Load file to calculate size
                tensors = load_file(file_path)
                total_size += sum(tensor.nbytes for tensor in tensors.values())

        update_safetensors_index(save_directory, total_size, weight_map)
        logger.info("Updated safetensors index")

    logger.info(f"Consolidation complete. Output saved to {save_directory}")
