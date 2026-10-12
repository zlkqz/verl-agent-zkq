#!/usr/bin/env bash

grpo_invalid_penalty_mode=separate
EXP_NAME_SUFFIX="_separate_penalty"

# All other settings, paths and launch behavior come from the main script.
source "$(dirname -- "${BASH_SOURCE[0]}")/run_webshop.sh" "$@"
