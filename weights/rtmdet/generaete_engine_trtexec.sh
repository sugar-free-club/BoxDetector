/usr/src/tensorrt/bin/trtexec \
        --onnx=./work_dirs/rtmdet/end2end.onnx \
        --saveEngine=./work_dirs/rtmdet/end2end_trtexec_fp32.plan \
        --workspace=100000000 \
        --buildOnly \
        --noTF32 \
        --verbose \