#!/usr/bin/env bash
# Kept with the benchmark because a recreated environment must prefer its own C++ runtime.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
