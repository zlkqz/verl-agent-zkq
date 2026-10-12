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

"""Agent GRPO grouping and diagnostics, independent of Ray/GPU workers.

`step/joint` preserves step-level reward normalization. `trajectory/joint`
normalizes episode task return + SUM of unique turn penalties, once per trajectory.
`separate` normalizes task returns only, then adds the raw local penalty to A.
Padding duplicates never contribute twice to group statistics.
"""
from collections import defaultdict

import numpy as np
import torch


ACTION_DIAGNOSTICS = (
    'is_action_valid', 'action_format_valid', 'action_syntax_valid',
    'action_executable', 'think_tags_complete', 'response_has_cjk',
    'think_open_count', 'think_close_count',
)


def unique_turn_indices(non_tensor):
    trajectories = non_tensor['traj_uid']
    turns = non_tensor.get('turn_index', np.arange(len(trajectories)))
    seen, result = set(), []
    for i, pair in enumerate(zip(trajectories, turns)):
        if pair not in seen:
            seen.add(pair)
            result.append(i)
    return result


def action_metrics(non_tensor, prefix='episode'):
    indices = unique_turn_indices(non_tensor)
    return {
        f'{prefix}/{key}/mean': float(np.asarray(non_tensor[key], dtype=float)[indices].mean())
        for key in ACTION_DIAGNOSTICS if key in non_tensor and indices
    }


def _summary(values, prefix):
    v = np.asarray(values, dtype=float)
    return {f'{prefix}/{k}': float(f(v)) for k, f in
            [('mean', np.mean), ('min', np.min), ('max', np.max),
             ('p10', lambda x: np.quantile(x, .1))]}


@torch.no_grad()
def compute_agent_grpo(task_rewards, penalties, response_mask, non_tensor,
                       grouping='step', penalty_mode='joint', normalize=True, epsilon=1e-6):
    if grouping not in ('step', 'trajectory') or penalty_mode not in ('joint', 'separate'):
        raise ValueError(f'Unknown GRPO grouping/penalty mode: {grouping}/{penalty_mode}')
    task = task_rewards.detach().cpu().double().numpy()
    penalty = penalties.detach().cpu().double().numpy()
    uid, traj = non_tensor['uid'], non_tensor['traj_uid']
    unique = unique_turn_indices(non_tensor)
    trajectory_rows, group_rows = defaultdict(list), defaultdict(list)
    for i in unique:
        trajectory_rows[(uid[i], traj[i])].append(i)
        group_rows[uid[i]].append(i)
    trajectory_task, trajectory_return = {}, {}
    for key, rows in trajectory_rows.items():
        if not np.allclose(task[rows], task[rows[0]], rtol=1e-5, atol=1e-6):
            raise ValueError('Agent GRPO requires the same episode task reward on every turn')
        trajectory_task[key] = task[rows[0]]
        trajectory_return[key] = task[rows[0]] + penalty[rows].sum()
    group_trajectories = defaultdict(list)
    for key in trajectory_rows:
        group_trajectories[key[0]].append(key)
    means, stds, task_stds, failures, zero_variance = {}, {}, [], [], []
    for group, rows in group_rows.items():
        tasks = np.array([trajectory_task[k] for k in group_trajectories[group]])
        task_stds.append(float(tasks.std(ddof=1)) if len(tasks) > 1 else 0.)
        failures.append(bool(np.all(tasks <= 0)))
        if grouping == 'trajectory':
            values = np.array([(trajectory_return if penalty_mode == 'joint' else trajectory_task)[k]
                               for k in group_trajectories[group]])
        else:
            values = task[rows] + (penalty[rows] if penalty_mode == 'joint' else 0.)
        means[group] = values.mean()
        stds[group] = values.std(ddof=1) if len(values) > 1 else 0.
        zero_variance.append(stds[group] <= epsilon)
    scalar = np.zeros(len(task), dtype=np.float64)
    for i in range(len(task)):
        if grouping == 'trajectory':
            value = (trajectory_return if penalty_mode == 'joint' else trajectory_task)[(uid[i], traj[i])]
        else:
            value = task[i] + (penalty[i] if penalty_mode == 'joint' else 0.)
        scalar[i] = (value - means[uid[i]]) / (stds[uid[i]] + epsilon) if normalize else value - means[uid[i]]
    task_advantage = scalar.copy()
    if penalty_mode == 'separate':
        scalar += penalty
    advantages = torch.as_tensor(scalar, dtype=response_mask.dtype if response_mask.is_floating_point()
                                 else torch.float32, device=response_mask.device).unsqueeze(-1) * response_mask
    metrics = {
        **_summary(task_stds, 'grpo/task_group_std'),
        **_summary(list(stds.values()), 'grpo/normalization_group_std'),
        'grpo/group_count': len(group_rows),
        'grpo/all_failed_group_ratio': float(np.mean(failures)),
        'grpo/zero_task_std_group_ratio': float(np.mean(np.asarray(task_stds) <= epsilon)),
        'grpo/zero_normalization_std_group_ratio': float(np.mean(zero_variance)),
        'grpo/unique_trajectories': len(trajectory_rows),
        'grpo/unique_turns': len(unique),
        'grpo/padding_duplicate_rows': len(task) - len(unique),
        'grpo/penalty/mean': float(penalty[unique].mean()),
        'grpo/penalty/abs_sum': float(np.abs(penalty[unique]).sum()),
        'grpo/normalized_advantage/abs_max': float(np.abs(task_advantage).max()),
    }
    return advantages, metrics


def advantage_metrics(advantages, response_mask, non_tensor):
    """Token counts describe actual optimizer input, including batch padding copies."""
    active = response_mask.bool()
    total = max(int(active.sum().item()), 1)
    metrics = {}
    for name, sign in [('positive', advantages > 0), ('negative', advantages < 0), ('zero', advantages == 0)]:
        selected = sign & active
        count = int(selected.sum().item())
        metrics[f'advantage/{name}_token_count'] = count
        metrics[f'advantage/{name}_token_fraction'] = count / total
        metrics[f'advantage/{name}_absolute_mass'] = float(advantages[selected].abs().sum().item())
    metrics['advantage/valid_token_count'] = int(active.sum().item())
    lengths = response_mask.sum(-1)
    for valid, label in [(True, 'valid_action'), (False, 'invalid_action')]:
        if 'is_action_valid' in non_tensor:
            rows = torch.as_tensor(np.asarray(non_tensor['is_action_valid'], dtype=bool) == valid,
                                   device=advantages.device)
            metrics[f'advantage/{label}_token_count'] = int(lengths[rows].sum().item())
    return metrics
