#!/usr/bin/env bash

mini_batch_size=64
grpo_reward_grouping=trajectory
EXP_NAME_SUFFIX="_trajectory_grpo"

# All other settings, paths and launch behavior come from the main script.
source "$(dirname -- "${BASH_SOURCE[0]}")/run_webshop.sh" "$@"
