#!/usr/bin/env bash
set -e
PORT=${PORT:-8082}

./generate_config.py

pip install -e $HOME/love_matcher_project

luigid --port "$PORT"
