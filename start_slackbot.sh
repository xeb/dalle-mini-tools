#!/bin/bash
source ./venv/bin/activate
python dalle_mini_tools/slackbot.py \
    --output_dir="./output" \
    --base_uri="https://dalle-mini-tools.xeb.ai/output"
