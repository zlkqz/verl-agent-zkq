# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

"""CPU-only regression coverage; load leaf modules without optional Ray/env dependencies."""
import ast
import importlib.util
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


algo = load('verl/trainer/ppo/agent_diagnostics.py')
projection = load('agent_system/environments/env_package/webshop/projection.py').webshop_projection


def sample(task=(10., 10., 0.), penalty=(0., -.1, 0.)):
    return (torch.tensor(task), torch.tensor(penalty), torch.ones(len(task), 3),
            {'uid': np.array(['g'] * len(task)), 'traj_uid': np.array(['a', 'a', 'b']),
             'turn_index': np.array([0, 1, 0])})


@pytest.mark.parametrize('text,valid', [
    ('<action>click[blue]</action>', True),
    ('选择蓝色。</think><action>click[blue]</action>', True),
    ('<think></think></think><action>click[blue]</action>', True),
    ('<action>CLICK[BLUE]</action>', True),
    ('<action>click[missing]</action>', False),
    ('<action>click[search]</action>', False),
    ('<action>search[green shoes]</action>', True),
    ('<action>search[ ]</action>', False),
    ('<action>nonsense</action>', False),
    ('<action>click[blue]</action><action>click[red]</action>', False),
    ('<action>click[blue]</action> </action>', False),
    ('<action>click[blue]', False),
    ('click[blue]', False),
    ('<action>click[blue] extra</action>', False),
])
def test_webshop_validity(text, valid):
    actions, valids, diagnostics = projection([text], [{'clickables': ['blue', 'search']}], True)
    assert valids == [valid]
    if not valid:
        assert actions == ['']  # No accidental action extracted from invalid text.
    assert diagnostics[0]['response_has_cjk'] == ('选择' in text)


def test_page_state_and_inputs_not_mutated():
    text = ['<action>click[blue]</action>']
    assert projection(text, [{'clickables': ['blue']}])[1] == [True]
    assert projection(text, [{'clickables': ['red']}])[1] == [False]
    assert text == ['<action>click[blue]</action>']


def test_prompt_does_not_force_thinking_tags():
    prompts = load('agent_system/environments/prompts/webshop.py')
    for prompt in [prompts.WEBSHOP_TEMPLATE, prompts.WEBSHOP_TEMPLATE_NO_HIS]:
        assert '<think>' not in prompt
        assert '<action>' in prompt


def test_joint_step_matches_original_grpo_without_padding():
    # Execute the existing production function without importing unrelated dependencies.
    tree = ast.parse((ROOT / 'verl/trainer/ppo/core_algos.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_grpo_outcome_advantage')
    namespace = dict(torch=torch, np=np, defaultdict=defaultdict)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<core_algos>', 'exec'), namespace)
    task, penalty, mask, non = sample()
    expected, _ = namespace['compute_grpo_outcome_advantage']((task + penalty)[:, None], mask, non['uid'], non['traj_uid'])
    actual, _ = algo.compute_agent_grpo(task, penalty, mask, non)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_trajectory_counts_outcome_once_and_sums_penalties():
    args = sample()
    actual, metrics = algo.compute_agent_grpo(*args, grouping='trajectory')
    expected = torch.tensor([1., 1., -1.]) / np.sqrt(2)
    torch.testing.assert_close(actual[:, 0], expected.float(), atol=1e-6, rtol=1e-5)
    assert metrics['grpo/normalization_group_std/mean'] == pytest.approx(9.9 / np.sqrt(2))
    assert metrics['grpo/task_group_std/mean'] == pytest.approx(10 / np.sqrt(2))


@pytest.mark.parametrize('grouping', ['step', 'trajectory'])
@pytest.mark.parametrize('penalty_mode', ['joint', 'separate'])
def test_padding_and_reordering_do_not_change_group_statistics(grouping, penalty_mode):
    task, penalty, mask, non = sample()
    before, old_metrics = algo.compute_agent_grpo(task, penalty, mask, non, grouping, penalty_mode)
    indices = np.array([2, 0, 1, 1, 0])
    after, metrics = algo.compute_agent_grpo(task[indices], penalty[indices], mask[indices],
                                           {k: v[indices] for k, v in non.items()}, grouping, penalty_mode)
    torch.testing.assert_close(after, before[indices])
    assert metrics['grpo/padding_duplicate_rows'] == 2
    assert metrics['grpo/task_group_std/mean'] == old_metrics['grpo/task_group_std/mean']
    assert metrics['grpo/normalization_group_std/mean'] == old_metrics['grpo/normalization_group_std/mean']


def test_separate_penalty_does_not_amplify_when_every_trajectory_fails():
    args = sample(task=(0., 0., 0.))
    advantages, metrics = algo.compute_agent_grpo(*args, penalty_mode='separate')
    torch.testing.assert_close(advantages[:, 0], torch.tensor([0., -.1, 0.]))
    assert metrics['grpo/all_failed_group_ratio'] == 1
    assert metrics['grpo/task_group_std/mean'] == 0


def test_constant_group_has_zero_advantage_and_no_nan():
    task, penalty, mask, non = sample(task=(0., 0., 0.), penalty=(0., 0., 0.))
    for grouping in ['step', 'trajectory']:
        a, _ = algo.compute_agent_grpo(task, penalty, mask, non, grouping)
        assert torch.isfinite(a).all() and not a.any()


def test_token_metrics_exclude_padding_and_count_signs():
    advantages = torch.tensor([[2., 2., 99.], [-3., 99., 99.], [0., 0., 99.]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 0]])
    metrics = algo.advantage_metrics(advantages, mask, {'is_action_valid': [True, False, True]})
    assert metrics['advantage/positive_token_count'] == 2
    assert metrics['advantage/negative_token_count'] == 1
    assert metrics['advantage/zero_token_count'] == 2
    assert metrics['advantage/negative_absolute_mass'] == 3
    assert metrics['advantage/invalid_action_token_count'] == 1


def test_trainer_routes_separate_penalties_without_double_subtraction():
    # Execute the actual routing function on tensor dictionaries, no Ray runtime.
    tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_advantage')
    namespace = dict(DataProto=object, torch=torch, compute_agent_grpo=algo.compute_agent_grpo,
                     AdvantageEstimator=SimpleNamespace(GAE='gae', GRPO='grpo'))
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<ray_trainer>', 'exec'), namespace)
    task, penalties, mask, non = sample(task=(0., 0., 0.))
    data = SimpleNamespace(batch={'task_rewards': task, 'invalid_action_penalties': penalties,
                                 'response_mask': mask, 'token_level_rewards': penalties[:, None]},
                           non_tensor_batch=non, meta_info={})
    result = namespace['compute_advantage'](data, 'grpo', grpo_invalid_penalty_mode='separate')
    torch.testing.assert_close(result.batch['advantages'][:, 0], penalties)
    assert 'grpo_diagnostics' in result.meta_info


def test_penalty_is_recorded_and_task_reward_is_unchanged():
    tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'apply_invalid_action_penalty')
    namespace = dict(DataProto=object, torch=torch, np=np)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<ray_trainer>', 'exec'), namespace)

    class Batch:
        def __init__(self):
            self.batch = {'token_level_scores': torch.tensor([[0., 10.], [0., 0.]]),
                          'prompts': torch.ones(2, 2, dtype=torch.long),
                          'attention_mask': torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
                          'task_rewards': torch.tensor([10., 0.])}
            self.non_tensor_batch = {'is_action_valid': np.array([True, False])}

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return SimpleNamespace(batch={k: v[i] for k, v in self.batch.items()},
                                   non_tensor_batch={k: v[i] for k, v in self.non_tensor_batch.items()})

    data, metrics = namespace['apply_invalid_action_penalty'](Batch(), .1)
    torch.testing.assert_close(data.batch['task_rewards'], torch.tensor([10., 0.]))
    torch.testing.assert_close(data.batch['invalid_action_penalties'], torch.tensor([0., -.1]))
    torch.testing.assert_close(data.batch['token_level_scores'], torch.tensor([[0., 10.], [-.1, 0.]]))
    assert metrics['episode/valid_action_ratio'] == .5


def test_manager_checks_pre_action_page_and_refreshes_page_state():
    tree = ast.parse((ROOT / 'agent_system/environments/env_manager.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WebshopEnvironmentManager')
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
    namespace = dict(List=list, to_numpy=np.asarray)
    exec(compile(ast.Module(body=[step], type_ignores=[]), '<env_manager>', 'exec'), namespace)
    actions_seen = []

    def env_step(actions):
        actions_seen.extend(actions)
        return ['next'], [0], [False], [{'available_actions': {'clickables': ['red']}}]

    manager = SimpleNamespace(
        available_actions=[{'clickables': ['blue']}], projection_f=projection,
        envs=SimpleNamespace(step=env_step), format_obs=lambda x: x,
        memory=SimpleNamespace(store=lambda x: None), pre_text_obs=['current'],
        build_text_obs=lambda obs, infos: obs)
    result = namespace['step'](manager, ['</think><action>click[blue]</action>'])
    assert actions_seen == ['click[blue]']
    assert result[3][0]['is_action_valid']
    assert manager.available_actions == [{'clickables': ['red']}]
    result = namespace['step'](manager, ['<action>click[blue]</action>'])
    assert actions_seen[-1] == ''
    assert not result[3][0]['is_action_valid']


@pytest.mark.parametrize('name,mini,grouping,penalty,entropy,kl', [
    ('trajectory_grpo', '256', 'trajectory', 'joint', '0.0', 'False'),
    ('separate_penalty', '256', 'step', 'separate', '0.0', 'False'),
    ('with_entropy', '256', 'step', 'joint', '0.001', 'False'),
    ('with_kl', '256', 'step', 'joint', '0.0', 'True'),
])
def test_independent_experiment_commands(name, mini, grouping, penalty, entropy, kl):
    import os
    import shlex
    import subprocess
    result = subprocess.run(['bash', str(ROOT / f'my_scripts/run_webshop_{name}.sh')],
                            env={**os.environ, 'DRY_RUN': '1'}, text=True, capture_output=True, check=True)
    args = dict(arg.split('=', 1) for arg in shlex.split(result.stdout.splitlines()[-1]) if '=' in arg)
    baseline = subprocess.run(['bash', str(ROOT / 'my_scripts/run_webshop.sh')],
                              env={**os.environ, 'DRY_RUN': '1'}, text=True, capture_output=True, check=True)
    base_args = dict(arg.split('=', 1) for arg in shlex.split(baseline.stdout.splitlines()[-1]) if '=' in arg)
    differences = {key for key in base_args.keys() | args.keys() if base_args.get(key) != args.get(key)}
    expected_differences = {'trainer.experiment_name', 'trainer.default_local_dir'} | {
        'trajectory_grpo': {'algorithm.grpo.reward_grouping'},
        'separate_penalty': {'algorithm.grpo.invalid_penalty_mode'},
        'with_entropy': {'actor_rollout_ref.actor.entropy_coeff'},
        'with_kl': {'actor_rollout_ref.actor.use_kl_loss', 'actor_rollout_ref.actor.kl_loss_coef'},
    }[name]
    assert differences == expected_differences
    experiment = args['trainer.experiment_name']
    expected_name = f'p2o_webshop_qwen3_8b_think_bs_16x8_mini_batch_{mini}_temperature0.8_length4096+4096_lr1e-6_{name}'
    assert experiment == expected_name
    assert args['trainer.default_local_dir'].endswith('/saved_models/p2o-agent/' + experiment)
    assert args['actor_rollout_ref.actor.ppo_mini_batch_size'] == mini
    assert args['algorithm.grpo.reward_grouping'] == grouping
    assert args['algorithm.grpo.invalid_penalty_mode'] == penalty
    assert args['actor_rollout_ref.actor.entropy_coeff'] == entropy
    assert args['actor_rollout_ref.actor.use_kl_loss'] == kl
    assert args['actor_rollout_ref.actor.invalid_action_penalty_coef'] == '0.1'
    assert args['algorithm.use_kl_in_reward'] == 'False'
    assert args['trainer.resume_mode'] == 'auto'
    if kl == 'True':
        assert args['actor_rollout_ref.actor.kl_loss_coef'] == '0.01'


def test_actor_tail_minibatch_uses_its_actual_gradient_accumulation():
    tree = ast.parse((ROOT / 'verl/workers/actor/dp_actor.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DataParallelPPOActor')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'update_policy')
    method.decorator_list = []

    class TensorBatch(dict):
        def __len__(self):
            return next(iter(self.values())).shape[0]

        def split(self, size):
            return [TensorBatch({k: v[i:i + size] for k, v in self.items()}) for i in range(0, len(self), size)]

        def to(self, device):
            return self

    class Config(dict):
        __getattr__ = dict.__getitem__

    def append(metrics, values):
        for key, value in values.items():
            metrics.setdefault(key, []).append(value)

    def loss(**kwargs):
        z = torch.tensor(0.)
        return -kwargs['log_prob'].mean(), z, z, z

    namespace = dict(DataProto=type('UnusedDataProto', (), {}), torch=torch,
                     append_to_dict=append, compute_policy_loss=loss,
                     get_torch_device=lambda: SimpleNamespace(current_device=lambda: 'cpu'))
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<dp_actor>', 'exec'), namespace)
    parameter = torch.tensor(1., requires_grad=True)
    gradients = []

    def optimizer_step():
        gradients.append(parameter.grad.item())
        return parameter.grad.abs().clone()

    actor = SimpleNamespace(
        actor_module=SimpleNamespace(train=lambda: None),
        actor_optimizer=SimpleNamespace(zero_grad=lambda: setattr(parameter, 'grad', None)),
        config=Config(ppo_mini_batch_size=8, ppo_micro_batch_size_per_gpu=4, use_dynamic_bsz=False,
                      use_kl_loss=False, ppo_epochs=1, clip_ratio=.2, clip_ratio_low=.2,
                      clip_ratio_high=.2, entropy_coeff=0., loss_agg_mode='token-mean',
                      policy_loss={'loss_mode': 'vanilla'}, grad_clip=1.),
        _optimizer_step=optimizer_step,
        _forward_micro_batch=lambda micro_batch, **kwargs: (None, parameter.expand_as(micro_batch['responses'])),
    )
    tensors = TensorBatch({k: torch.ones(12, 2) for k in
                          ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']})
    data = SimpleNamespace(batch=tensors, non_tensor_batch={}, meta_info={'temperature': 1.},
                           select=lambda **kwargs: SimpleNamespace(batch=tensors))
    metrics = namespace['update_policy'](actor, data)
    assert gradients == pytest.approx([-1., -1.])  # Previously the tail gradient was -0.5.
    assert metrics['actor/optimizer_steps'] == [2]
    assert metrics['actor/entropy_bonus'] == [0., 0., 0.]
