"""Reasoning-prefix, prompt-format and action-validity regression coverage."""

import importlib.util
import os
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chat = load("verl/utils/agent_chat_template.py")
alfworld = load("agent_system/environments/env_package/alfworld/projection.py").alfworld_projection
webshop = load("agent_system/environments/env_package/webshop/projection.py").webshop_projection


@pytest.mark.parametrize("enable_thinking", [None, True, False])
@pytest.mark.parametrize("require_think", [False, True])
def test_independent_prompt_and_template_options(enable_thinking, require_think):
    options = {"enable_thinking": enable_thinking, "custom_option": "retained"}
    config = {"prompt_require_think": require_think, "apply_chat_template_kwargs": options}
    expected = {"custom_option": "retained"}
    if enable_thinking is not None:
        expected["enable_thinking"] = enable_thinking
    assert chat.get_chat_template_kwargs(config) == expected
    assert config["apply_chat_template_kwargs"] == options

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"add_generation_prompt": True, "tokenize": False, **expected}
            return "model template output"

    # Pass through the model's output without imposing a Qwen-specific suffix.
    assert chat.apply_generation_chat_template(Tokenizer(), [], config) == "model template output"


def test_missing_enable_thinking_uses_template_default():
    assert chat.get_chat_template_kwargs({}) == {}
    assert chat.get_chat_template_kwargs({"apply_chat_template_kwargs": {"custom": 1}}) == {"custom": 1}


@pytest.mark.parametrize("prefix", ["<think>reason</think>", "reason</think>",
                                   "reason without tags. ", "<think>\n\n</think>\n\nreason. ", ""])
def test_action_validity_does_not_depend_on_reasoning_format(prefix):
    assert alfworld([prefix + "<action>look</action>"], [["look"]]) == (["look"], [True])
    assert webshop([prefix + "<action>click[blue]</action>"], [{"clickables": ["blue"]}]) == (["click[blue]"], [True])
    assert alfworld([prefix + "<action>teleport</action>"], [["look"]]) == ([""], [False])
    assert webshop([prefix + "<action>click[missing]</action>"], [{"clickables": ["blue"]}]) == ([""], [False])


def launch(script, **settings):
    env = {k: v for k, v in os.environ.items() if k not in ("PROMPT_REQUIRE_THINK", "ENABLE_THINKING", "MODEL_PATH", "EXP_NAME", "EXP_NAME_SUFFIX")}
    result = subprocess.run(["bash", str(ROOT / "my_scripts" / script), "vllm", "trainer.total_epochs=3"],
                            env={**env, "DRY_RUN": "1", **settings}, text=True, capture_output=True, check=True)
    return dict(arg.split("=", 1) for arg in shlex.split(result.stdout.splitlines()[-1]) if "=" in arg)


@pytest.mark.parametrize("task", ["alfworld", "webshop"])
def test_launch_controls_names_and_dedicated_non_thinking_script(task):
    default = launch(f"run_{task}.sh")
    assert default["data.prompt_require_think"] == "false"
    assert default["data.apply_chat_template_kwargs.enable_thinking"] == "null"
    assert f"p2o_{task}_qwen3_8b_think_" in default["trainer.experiment_name"]
    for prompt in ["true", "false"]:
        for enabled in ["null", "true", "false"]:
            args = launch(f"run_{task}.sh", PROMPT_REQUIRE_THINK=prompt, ENABLE_THINKING=enabled)
            assert args["data.prompt_require_think"] == prompt
            assert args["data.apply_chat_template_kwargs.enable_thinking"] == enabled
            assert args["trainer.experiment_name"] == default["trainer.experiment_name"]
    dedicated = launch(f"run_{task}_non_think.sh")
    assert dedicated["data.prompt_require_think"] == "false"
    assert dedicated["data.apply_chat_template_kwargs.enable_thinking"] == "false"
    assert f"p2o_{task}_qwen3_8b_non_think_" in dedicated["trainer.experiment_name"]
    assert dedicated["trainer.experiment_name"] in dedicated["trainer.default_local_dir"]
    instruct = launch(f"run_{task}.sh", PROMPT_REQUIRE_THINK="true", EXP_NAME="example_instruct_experiment", MODEL_PATH="/models/instruct")
    assert instruct["data.prompt_require_think"] == "true"
    assert instruct["data.apply_chat_template_kwargs.enable_thinking"] == "null"
    assert instruct["actor_rollout_ref.model.path"] == "/models/instruct"
    assert instruct["trainer.experiment_name"] == "example_instruct_experiment"
    assert instruct["trainer.total_epochs"] == "3"
    for script in [f"run_{task}.sh", f"run_{task}_non_think.sh", f"run_{task}_with_kl.sh"]:
        custom = launch(script, EXP_NAME="custom_experiment", EXP_NAME_SUFFIX="_unused_suffix")
        assert custom["trainer.experiment_name"] == "custom_experiment"
        assert custom["trainer.default_local_dir"].endswith("/custom_experiment")
    assert launch(f"run_{task}.sh", EXP_NAME="")["trainer.experiment_name"] == default["trainer.experiment_name"]


@pytest.mark.parametrize("require_think,enable_thinking,model_kind", [
    (prompt, enabled, "qwen3") for prompt in [False, True] for enabled in [None, True, False]
] + [(True, None, "instruct")])
def test_real_collector_and_dataset(require_think, enable_thinking, model_kind, tmp_path):
    """Opt-in CPU integration with the model tokenizer and production call paths."""
    model_path = os.environ.get("TEST_QWEN3_TOKENIZER_PATH" if model_kind == "qwen3" else "TEST_INSTRUCT_TOKENIZER_PATH")
    if not model_path:
        pytest.skip("Set the local tokenizer path for this model")
    from datasets import Dataset
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from types import SimpleNamespace
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from verl.utils.dataset.rl_dataset import RLHFDataset

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = OmegaConf.create({"data": {"prompt_require_think": require_think,
                                       "apply_chat_template_kwargs": {"enable_thinking": enable_thinking},
                                       "max_prompt_length": 512,
                                       "truncation": "error", "return_raw_chat": True,
                                       "filter_overlong_prompts": True, "filter_overlong_prompts_workers": 1}})
    collector = TrajectoryCollector(config, tokenizer)
    batch = SimpleNamespace(non_tensor_batch={"raw_prompt": [[{"role": "user", "content": ""}]],
                                             "data_source": ["text"]})
    for task, action in [("alfworld", "look"), ("webshop", "click[blue]")]:
        templates = load(f"agent_system/environments/prompts/{task}.py")
        for name in [f"{task.upper()}_TEMPLATE_NO_HIS", f"{task.upper()}_TEMPLATE"]:
            text = getattr(templates, name).format(
                current_observation="test", task_description="test", admissible_actions="look",
                available_actions="click[blue]", step_count=1, current_step=2,
                history_length=1, action_history="test",
            )
            row = collector.preprocess_single_sample(0, batch, {"text": [text]})
            content = row["raw_prompt"][0]["content"]
            assert (chat.REASONING_TAG_INSTRUCTION in content) == require_think
            prompt = tokenizer.decode(row["raw_prompt_ids"], skip_special_tokens=False)
            assert row["input_ids"][row["attention_mask"].bool()].tolist() == row["raw_prompt_ids"]
            if model_kind == "qwen3" and enable_thinking is False:
                assert prompt.endswith("<think>\n\n</think>\n\n")
            generated = f"I choose this action. <action>{action}</action>"
            if enable_thinking is not False:
                generated = "<think>I choose this action.</think>" + f"<action>{action}</action>"
            elif require_think:
                generated = "<think>I choose this action.</think>" + f"<action>{action}</action>"
            if task == "alfworld":
                assert alfworld([generated], [["look"]])[1] == [True]
            else:
                assert webshop([generated], [{"clickables": ["blue"]}])[1] == [True]

    parquet = tmp_path / "test.parquet"
    Dataset.from_dict({"prompt": [[{"role": "user", "content": "test"}]]}).to_parquet(str(parquet))
    dataset = RLHFDataset(str(parquet), tokenizer, config.data)
    expected = chat.apply_generation_chat_template(tokenizer, [{"role": "user", "content": "test"}], config.data)
    item = dataset[0]
    ids = item["input_ids"][item["attention_mask"].bool()].tolist()
    assert ids == tokenizer.encode(expected, add_special_tokens=False)
    # Filtering must account for the same prefix, including the empty think block.
    short_config = OmegaConf.merge(config.data, {"max_prompt_length": len(ids) - 1})
    assert len(RLHFDataset(str(parquet), tokenizer, short_config)) == 0
