/usr/src/tensorrt/bin/trtexec \
        --onnx=./work_dirs/rtmdet/end2end_simplified.onnx \
        --saveEngine=./work_dirs/rtmdet/end2end_trtexec_fp16_simplified.plan \
        --workspace=100000000 \
        --buildOnly \
        --fp16 \
        --verbose \