/usr/src/tensorrt/bin/trtexec \
        --onnx=./work_dirs/rtmdet/end2end_simplified.onnx \
        --saveEngine=./work_dirs/rtmdet/end2end_trtexec_fp32_simplified.plan \
        --workspace=100000000 \
        --buildOnly \
        --noTF32 \
        --verbose \