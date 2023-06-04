#!/usr/bin/env bash
python3.7 projects/easydeploy/tools/build_engine.py \
    work_dirs/rtmdet/end2end_simplified.onnx \
    --img-size 640 640 \
    --device cuda:0


