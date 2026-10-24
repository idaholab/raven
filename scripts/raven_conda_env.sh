#!/usr/bin/env bash
set -e

# Wrapper for CIVET recipes to create/load the RAVEN conda env.
if [[ "${DEFAULT_INSTALL:-0}" == "1" ]]; then
  ./scripts/establish_conda_env.sh --install
else
  ./scripts/establish_conda_env.sh --load
fi

# Ensure conda libstdc++ is preferred on Linux runners.
if [[ "$(uname -s)" == "Linux" && -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
