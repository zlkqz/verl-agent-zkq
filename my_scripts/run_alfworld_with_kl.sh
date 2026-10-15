#!/usr/bin/env bash

use_kl_loss=True
kl_loss_coef=0.01
EXP_NAME_SUFFIX="_with_kl"

# All other settings, paths and launch behavior come from the main script.
source "$(dirname -- "${BASH_SOURCE[0]}")/run_alfworld.sh" "$@"
