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

"""Exercise the trainer save/load methods on CPU without Ray or GPU workers."""

import ast
import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("retention", ROOT / "verl/utils/checkpoint/retention.py")
retention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retention)


def load_trainer_methods():
    namespace = dict(os=os, torch=torch, **{name: getattr(retention, name) for name in (
        "has_hf_weights", "validate_fsdp_checkpoint", "update_checkpoint_tracker", "prune_old_training_states",
    )})
    # Use the production tracker lookup without importing its GPU dependencies.
    tree = ast.parse((ROOT / "verl/utils/checkpoint/checkpoint_manager.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("find_latest_ckpt_path", "get_checkpoint_tracker_filename")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<checkpoint_manager>", "exec"), namespace)
    tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    nodes = [node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name in ("_save_checkpoint", "_load_checkpoint")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<ray_trainer>", "exec"), namespace)
    return namespace


def write(path, content="saved"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def make_role(path, hf=True, world_size=2):
    for rank in range(world_size):
        for kind in ("model", "optim", "extra_state"):
            write(path / f"{kind}_world_size_{world_size}_rank_{rank}.pt")
    write(path / "config.json", "{}")
    write(path / "tokenizer.json", "{}")
    if hf:
        write(path / "huggingface/config.json", "{}")
        write(path / "huggingface/tokenizer.json", "{}")
        write(path / "huggingface/model.safetensors")


def make_trainer(root, *, failure=None, use_critic=False, keep_latest=True, previous_step=120):
    methods = load_trainer_methods()
    config = OmegaConf.create({
        "trainer": {
            "default_local_dir": str(root), "default_hdfs_dir": None,
            "keep_latest_full_checkpoint": keep_latest,
            "max_actor_ckpt_to_keep": 1, "max_critic_ckpt_to_keep": 1,
            "resume_mode": "auto", "del_local_ckpt_after_load": False,
        },
        "actor_rollout_ref": {"actor": {"strategy": "fsdp", "checkpoint": {"contents": ["model", "optimizer", "extra", "hf_model"]}}},
        "critic": {"strategy": "fsdp", "checkpoint": {"contents": ["model", "optimizer", "extra", "hf_model"]}},
    })
    old = root / f"global_step_{previous_step}/actor/optim_world_size_2_rank_0.pt"
    events = []

    class Worker:
        world_size = 2

        def __init__(self, role):
            self.role = role

        def save_checkpoint(self, path, remote, step, max_ckpt_to_keep=None):
            if keep_latest:
                assert max_ckpt_to_keep is None
            assert old.is_file(), "old state deleted before all save stages completed"
            events.append(self.role)
            make_role(Path(path))
            if failure == self.role:
                raise OSError("simulated worker save failure")
            if failure == "hf" and self.role == "actor":
                (Path(path) / "huggingface/model.safetensors").unlink()
            if failure == "shard" and self.role == "actor":
                (Path(path) / "optim_world_size_2_rank_1.pt").unlink()

        def load_checkpoint(self, path, del_local_after_load=False):
            assert not del_local_after_load
            assert Path(path, "optim_world_size_2_rank_0.pt").is_file()
            events.append(("load", self.role, path))

    class Dataloader:
        loaded = None

        def state_dict(self):
            assert old.is_file()
            assert (root / "latest_checkpointed_iteration.txt").read_text() == str(previous_step)
            events.append("dataloader")
            if failure == "dataloader":
                raise OSError("simulated dataloader failure")
            return {"position": previous_step + 20}

        def load_state_dict(self, state):
            self.loaded = state

    trainer = SimpleNamespace(config=config, global_steps=previous_step + 20, use_critic=use_critic,
                              actor_rollout_wg=Worker("actor"), critic_wg=Worker("critic"), train_dataloader=Dataloader())
    return trainer, methods, events


def seed_old_checkpoints(root, use_critic=False):
    for step in (20, 40, 60, 80, 100, 120):
        path = root / f"global_step_{step}"
        make_role(path / "actor", hf=False)
        if use_critic:
            make_role(path / "critic", hf=False)
        torch.save({"position": step}, path / "data.pt")
    write(root / "latest_checkpointed_iteration.txt", "120")


def test_resume_120_then_save_140_and_160(tmp_path):
    seed_old_checkpoints(tmp_path, use_critic=True)
    trainer, methods, events = make_trainer(tmp_path, use_critic=True)
    methods["_load_checkpoint"](trainer)
    assert trainer.global_steps == 120
    assert trainer.train_dataloader.loaded == {"position": 120}
    assert all(event[2].endswith(f"global_step_120/{event[1]}") for event in events)
    assert (tmp_path / "global_step_120/data.pt").is_file()
    trainer.global_steps = 140
    methods["_save_checkpoint"](trainer)
    assert events[-3:] == ["actor", "critic", "dataloader"]
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "140"
    for step in (20, 40, 60, 80, 100, 120):
        assert not (tmp_path / f"global_step_{step}/data.pt").exists()
        for role in ("actor", "critic"):
            path = tmp_path / f"global_step_{step}" / role
            assert (path / "model_world_size_2_rank_0.pt").read_text() == "saved"
            assert (path / "tokenizer.json").is_file()
            assert not list(path.glob("optim_*.pt"))
            assert not list(path.glob("extra_state_*.pt"))
    for role in ("actor", "critic"):
        retention.validate_fsdp_checkpoint(tmp_path / "global_step_140" / role, 2)
        assert retention.has_hf_weights(tmp_path / "global_step_140" / role / "huggingface")
    # A fresh process must also discover and prune 140 after committing 160.
    trainer, methods, _ = make_trainer(tmp_path, use_critic=True, previous_step=140)
    methods["_save_checkpoint"](trainer)
    for role in ("actor", "critic"):
        path = tmp_path / "global_step_140" / role
        assert {item.name for item in path.iterdir()} == {"huggingface"}
    retention.validate_fsdp_checkpoint(tmp_path / "global_step_160/actor", 2)
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "160"


@pytest.mark.parametrize("failure", ["actor", "critic", "dataloader", "hf", "shard", "data_write", "tracker"])
def test_failed_save_never_prunes_previous_checkpoint(tmp_path, monkeypatch, failure):
    seed_old_checkpoints(tmp_path, use_critic=True)
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    trainer, methods, _ = make_trainer(tmp_path, use_critic=True, failure=failure)

    def fail(*args, **kwargs):
        raise OSError("simulated commit failure")

    if failure == "data_write":
        monkeypatch.setattr(torch, "save", fail)
    if failure == "tracker":
        monkeypatch.setattr(retention.os, "replace", fail)
    with pytest.raises((OSError, RuntimeError)):
        methods["_save_checkpoint"](trainer)
    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
    assert not list(tmp_path.glob(".latest_checkpointed_iteration.*"))


def test_retention_disabled_keeps_original_behavior(tmp_path):
    seed_old_checkpoints(tmp_path)
    trainer, methods, _ = make_trainer(tmp_path, keep_latest=False)
    methods["_save_checkpoint"](trainer)
    assert (tmp_path / "global_step_120/actor/optim_world_size_2_rank_0.pt").is_file()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "140"


@pytest.mark.parametrize("missing", ["model", "optimizer", "extra", "hf_model"])
def test_incompatible_contents_fail_before_any_write(tmp_path, missing):
    seed_old_checkpoints(tmp_path)
    trainer, methods, events = make_trainer(tmp_path)
    trainer.config.actor_rollout_ref.actor.checkpoint.contents.remove(missing)
    with pytest.raises(ValueError):
        methods["_save_checkpoint"](trainer)
    assert not events
    assert not (tmp_path / "global_step_140").exists()


def test_incomplete_hf_export_keeps_legacy_model_shards(tmp_path):
    path = tmp_path / "global_step_120/actor"
    make_role(path, hf=False)
    write(path / "huggingface/config.json", "{}")
    write(path / "huggingface/model.safetensors.index.json", json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}))
    write(path / "huggingface/model-1.safetensors")
    assert not retention.has_hf_weights(path / "huggingface")
    retention.prune_old_training_states(tmp_path, 140)
    assert (path / "model_world_size_2_rank_0.pt").is_file()
    write(path / "huggingface/model-2.safetensors")
    assert retention.has_hf_weights(path / "huggingface")
    retention.prune_old_training_states(tmp_path, 140)
    assert {item.name for item in path.iterdir()} == {"huggingface"}


def test_cleanup_does_not_touch_latest_future_or_unrelated_paths(tmp_path):
    for name in ("global_step_140", "global_step_160", "global_step_backup", "other"):
        make_role(tmp_path / name / "actor")
    external = tmp_path / "other"
    (tmp_path / "global_step_20").symlink_to(external, target_is_directory=True)
    retention.prune_old_training_states(tmp_path, 140)
    for name in ("global_step_140", "global_step_160", "global_step_backup", "other"):
        retention.validate_fsdp_checkpoint(tmp_path / name / "actor", 2)


@pytest.mark.parametrize("script", sorted((ROOT / "my_scripts").glob("run_*.sh")), ids=lambda path: path.name)
def test_all_training_scripts_enable_retention_and_allow_resume_override(script):
    result = subprocess.run(["bash", str(script), "vllm"], env={**os.environ, "DRY_RUN": "1"}, text=True, capture_output=True, check=True)
    args = shlex.split(result.stdout.splitlines()[-1])
    overrides = dict(arg.split("=", 1) for arg in args[4:])
    assert overrides["trainer.resume_mode"] == "auto"
    assert overrides["trainer.keep_latest_full_checkpoint"] == "True"
    for key in ("actor_rollout_ref.actor.checkpoint.contents", "critic.checkpoint.contents"):
        assert overrides[key] == "[model,optimizer,extra,hf_model]"
    result = subprocess.run(["bash", str(script), "vllm", "trainer.resume_mode=disable"], env={**os.environ, "DRY_RUN": "1"}, text=True, capture_output=True, check=True)
    args = shlex.split(result.stdout.splitlines()[-1])
    assert [arg for arg in args if arg.startswith("trainer.resume_mode=")][-1] == "trainer.resume_mode=disable"
