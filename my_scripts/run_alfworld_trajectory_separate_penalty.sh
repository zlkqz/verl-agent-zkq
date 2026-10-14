#!/usr/bin/env bash

mini_batch_size=256
grpo_reward_grouping=trajectory
grpo_invalid_penalty_mode=separate
EXP_NAME_SUFFIX="_trajectory_separate_penalty"

# Compare against separate_penalty with the same minibatch and local penalty.
source "$(dirname -- "${BASH_SOURCE[0]}")/run_alfworld.sh" "$@"
