#!/usr/bin/env bash
# Local (non-Docker) setup for the CW-Net toy demo.
# Requires swig for building box2d: sudo apt-get install swig
set -e
python3 -m venv cwnet_demo
source cwnet_demo/bin/activate
pip3 install --upgrade pip
pip install -r "$(dirname "$0")/requirements.txt"
