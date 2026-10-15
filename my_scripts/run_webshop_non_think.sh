#!/usr/bin/env bash

# Qwen3-8B: prefill an empty think block, then generate reasoning and action.
PROMPT_REQUIRE_THINK=false
ENABLE_THINKING=false
EXP_NAME=${EXP_NAME:-p2o_webshop_qwen3_8b_non_think_bs_16x8_mini_batch_256_temperature0.8_length4096+4096_lr1e-6}
source "$(dirname -- "${BASH_SOURCE[0]}")/run_webshop.sh" "$@"
