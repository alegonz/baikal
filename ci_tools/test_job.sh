#!/usr/bin/env bash
set -e

SKLEARN_VERSION=${1}

# Run from repo root directory
apt update && apt install -y graphviz
make setup_dev SKLEARN_VERSION=${SKLEARN_VERSION}

if [[ "${RUN_TEST_WITH_COVERAGE}" == "1" ]]; then
    make test-cov
    make upload-cov
else
    make test
fi

make type-check
