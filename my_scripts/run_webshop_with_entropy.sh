#!/usr/bin/env bash

entropy_coeff=0.001
EXP_NAME_SUFFIX="_with_entropy"

# All other settings, paths and launch behavior come from the main script.
source "$(dirname -- "${BASH_SOURCE[0]}")/run_webshop.sh" "$@"
