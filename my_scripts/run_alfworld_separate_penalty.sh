#!/usr/bin/env bash

mini_batch_size=256
grpo_reward_grouping=step
grpo_invalid_penalty_mode=separate
EXP_NAME_SUFFIX="_separate_penalty"

# Keep the minibatch fixed when comparing penalty normalization.
source "$(dirname -- "${BASH_SOURCE[0]}")/run_alfworld.sh" "$@"
