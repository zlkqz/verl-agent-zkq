#!/usr/bin/env bash

# Qwen3-8B: prefill an empty think block, then generate reasoning and action.
export PROMPT_REQUIRE_THINK=false
export ENABLE_THINKING=false
export max_response_length='512'
export val_before_train='True'
export micro_bs_per_gpu='32'
export max_steps='50'

EXP_NAME_PREFIX=${EXP_NAME_PREFIX:-p2o_alfworld_qwen3_8b_non_think}
source "$(dirname -- "${BASH_SOURCE[0]}")/run_alfworld.sh" "$@"
