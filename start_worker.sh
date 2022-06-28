#!/bin/bash
python dalle_mini_tools/worker.py \
    --postprocess="./postprocess.sh" \
    --postprocess_cwd="./" \
    --output_dir="./output"
