#!/usr/bin/env bash
python3.7 projects/easydeploy/tools/build_engine.py \
    work_dirs/yolox/end2end.onnx \
    --img-size 416 416 \
    --device cuda:0


# /usr/src/tensorrt/bin/trtexec \
#         --onnx=end2end_box.onnx \
#         --saveEngine=rtmdet_box.plan \
#         --workspace=40960 \
#         --buildOnly \
#         --noTF32 \
#         --verbose \