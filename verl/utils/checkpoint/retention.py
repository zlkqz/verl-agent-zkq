# Copyright 2026 verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keep historical inference weights and only the latest training checkpoint."""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path


def has_hf_weights(path):
    """Check for a complete HF weight export, rather than a config-only directory."""
    path = Path(path)
    if not (path / "config.json").is_file():
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (path / name).is_file() and (path / name).stat().st_size > 0:
            return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / name
        if not index.is_file():
            continue
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if shards and all((path / shard).is_file() and (path / shard).stat().st_size > 0 for shard in shards):
                return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return False


def update_checkpoint_tracker(root, step):
    """Publish the new step atomically, after all checkpoint writes have finished."""
    root = Path(root)
    with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix=".latest_checkpointed_iteration.", delete=False) as tracker:
        temporary_path = tracker.name
        tracker.write(str(step))
        tracker.flush()
        os.fsync(tracker.fileno())
    try:
        os.replace(temporary_path, root / "latest_checkpointed_iteration.txt")
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def validate_fsdp_checkpoint(path, world_size):
    """Require every rank's model, optimizer and scheduler/RNG shard on disk."""
    path = Path(path)
    for rank in range(world_size):
        for kind in ("model", "optim", "extra_state"):
            shard = path / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
            if not shard.is_file() or shard.stat().st_size == 0:
                raise RuntimeError(f"Incomplete checkpoint: missing or empty {shard}; previous checkpoint retained")


def _remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def prune_old_training_states(root, latest_step):
    """Prune on-disk older steps, including checkpoints saved before this run.

    HF exports are retained for each role. Legacy FSDP checkpoints without HF
    weights keep model shards and metadata so they can still be merged with
    scripts/model_merger.py; only optimizer/extra/dataloader states are removed.
    Newer steps are untouched when resuming an explicitly selected older step.
    """
    for step_path in Path(root).iterdir():
        match = re.fullmatch(r"global_step_(\d+)", step_path.name)
        if not match or int(match.group(1)) >= latest_step or not step_path.is_dir() or step_path.is_symlink():
            continue
        for role in ("actor", "critic"):
            role_path = step_path / role
            if not role_path.is_dir() or role_path.is_symlink():
                continue
            if has_hf_weights(role_path / "huggingface"):
                for item in role_path.iterdir():
                    if item.name not in ("huggingface", "lora_adapter"):
                        _remove(item)
            else:
                # A legacy huggingface/ directory may contain only config and
                # tokenizer files. Never discard its only model weight shards.
                for item in role_path.iterdir():
                    if re.fullmatch(r"(?:optim|extra_state)_world_size_\d+_rank_\d+\.pt", item.name) or item.name in ("optim", "rng_states"):
                        _remove(item)
        data_path = step_path / "data.pt"
        if data_path.exists():
            _remove(data_path)
        print(f"Pruned training states from {step_path}; preserved inference weights and metadata")
