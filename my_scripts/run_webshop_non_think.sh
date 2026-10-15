#!/usr/bin/env bash

# Qwen3-8B: prefill an empty think block, then generate reasoning and action.
PROMPT_REQUIRE_THINK=false
ENABLE_THINKING=false
# Override defaults here or through environment variables before launching.
val_data_size=${val_data_size:-128}
max_response_length=${max_response_length:-4096}
EXP_NAME_PREFIX=${EXP_NAME_PREFIX:-p2o_webshop_qwen3_8b_non_think}
source "$(dirname -- "${BASH_SOURCE[0]}")/run_webshop.sh" "$@"
