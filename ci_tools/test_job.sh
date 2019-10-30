#!/usr/bin/env bash
set -e

# Run from repo root directory
apt update && apt install -y graphviz
make setup_dev

if [[ "${RUN_TEST_WITH_COVERAGE}" == "1" ]]; then
    make test-cov
    make upload-cov
else
    make test
fi

make type-check
