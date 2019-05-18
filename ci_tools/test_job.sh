#!/usr/bin/env bash

# Run from repo root directory
apt update && apt install -y graphviz
make setup_dev
make test
