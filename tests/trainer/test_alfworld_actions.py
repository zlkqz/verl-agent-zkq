"""CPU coverage for ALFWorld action validation and experiment entry points."""
import ast
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'alfworld_projection', ROOT / 'agent_system/environments/env_package/alfworld/projection.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
projection = module.alfworld_projection


@pytest.mark.parametrize('text,valid', [
    ('<action>look</action>', True),
    ('中文推理。</think><action>look</action>', True),
    ('<think></think></think><action>look</action>', True),
    ('<ACTION>  LOOK  </ACTION>', True),
    ('<think>Check</think><action>teleport to mars</action>', False),
    ('<action>take desklamp 1 from desk 1</action>', False),
    ('<action>look</action><action>inventory</action>', False),
    ('<think>Could choose <action>inventory</action></think><action>look</action>', False),
    ('<action>look</action></action>', False),
    ('<action>look', False),
    ('</action><action>look</action>', False),
    ('<action></action>', False),
    ('<action>look\ninventory</action>', False),
    ('<action>look\x00</action>', False),
    ('<action><action>look</action></action>', False),
    ('look', False),
])
def test_action_validity(text, valid):
    actions, valids, diagnostics = projection([text], [['look', 'inventory']], True)
    assert valids == [valid]
    assert actions == ['look' if valid else '']
    assert diagnostics[0]['response_has_cjk'] == ('中文' in text)
    assert diagnostics[0]['action_executable'] == valid


def test_inputs_not_mutated_and_current_command_pool_is_authoritative():
    responses = ['<action>open cabinet 1</action>']
    pool = [['open cabinet 1']]
    assert projection(responses, pool) == (['open cabinet 1'], [True])
    assert projection(responses, [['close cabinet 1']]) == ([''], [False])
    assert responses == ['<action>open cabinet 1</action>']
    assert pool == [['open cabinet 1']]
    assert projection(responses, [None]) == ([''], [False])
    with pytest.raises(ValueError):
        projection(responses, [])


def test_command_spelling_from_environment_is_preserved():
    assert projection(['<action>look</action>'], [['LOOK']]) == (['LOOK'], [True])


def test_manager_validates_pre_action_pool_and_propagates_diagnostics():
    tree = ast.parse((ROOT / 'agent_system/environments/env_manager.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AlfWorldEnvironmentManager')
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
    namespace = dict(List=list, to_numpy=np.asarray)
    exec(compile(ast.Module(body=[step], type_ignores=[]), '<env_manager>', 'exec'), namespace)
    actions_seen = []
    env = SimpleNamespace(get_admissible_commands=[['open cabinet 1']])

    def env_step(actions):
        actions_seen.extend(actions)
        env.get_admissible_commands = [['close cabinet 1']]
        return ['next'], None, [0], [False], [{'extra.gamefile': 'test'}]

    env.step = env_step
    manager = SimpleNamespace(
        envs=env, projection_f=projection,
        memory=SimpleNamespace(store=lambda value: None), pre_text_obs=['current'],
        build_text_obs=lambda obs, pools: obs,
    )
    result = namespace['step'](manager, ['中文<action>open cabinet 1</action>'])
    assert actions_seen == ['open cabinet 1']
    assert result[3][0]['is_action_valid']
    assert result[3][0]['action_executable']
    assert result[3][0]['response_has_cjk']
    assert not result[3][0]['think_tags_complete']
    result = namespace['step'](manager, ['<action>open cabinet 1</action>'])
    assert actions_seen[-1] == ''
    assert not result[3][0]['is_action_valid']
    assert not result[3][0]['action_executable']


def launch_args(script, *overrides):
    result = subprocess.run(
        ['bash', str(ROOT / 'my_scripts' / script), 'vllm', *overrides],
        env={**os.environ, 'DRY_RUN': '1'}, cwd='/tmp',
        text=True, capture_output=True, check=True,
    )
    return dict(arg.split('=', 1) for arg in shlex.split(result.stdout.splitlines()[-1]) if '=' in arg)


def test_main_and_trajectory_experiment_settings():
    base = launch_args('run_alfworld.sh')
    trajectory = launch_args('run_alfworld_trajectory_grpo.sh')
    for args, mini, grouping, suffix in [(base, '256', 'step', ''),
                                        (trajectory, '256', 'trajectory', '_trajectory_grpo')]:
        name = f'p2o_alfworld_qwen3_8b_think_bs_16x8_mini_batch_{mini}_temperature0.8_length3072+3072_lr1e-6{suffix}'
        assert args['trainer.experiment_name'] == name
        assert args['trainer.default_local_dir'].endswith('/saved_models/p2o-agent/' + name)
        assert args['data.train_files'] == str(ROOT / 'data/alfworld/text/train.parquet')
        assert args['actor_rollout_ref.actor.ppo_mini_batch_size'] == mini
        assert args['actor_rollout_ref.actor.entropy_coeff'] == '0.0'
        assert args['actor_rollout_ref.actor.use_kl_loss'] == 'False'
        assert args['algorithm.use_kl_in_reward'] == 'False'
        assert args['algorithm.grpo.reward_grouping'] == grouping
        assert args['algorithm.grpo.invalid_penalty_mode'] == 'joint'
        assert args['trainer.resume_mode'] == 'auto'
    differences = {key for key in base if base[key] != trajectory[key]}
    assert differences == {'algorithm.grpo.reward_grouping', 'trainer.experiment_name',
                           'trainer.default_local_dir'}
    assert launch_args('run_alfworld_trajectory_grpo.sh', 'trainer.total_epochs=3')['trainer.total_epochs'] == '3'
