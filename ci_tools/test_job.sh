#!/usr/bin/env bash
set -ue

# Run from repo root directory
apt update && apt install -y graphviz
make setup_dev
make test-cov
codecov --token=${CODECOV_TOKEN}
