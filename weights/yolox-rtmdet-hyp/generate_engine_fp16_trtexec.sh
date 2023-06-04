/usr/src/tensorrt/bin/trtexec \
        --onnx=./work_dirs/yolox_rtmdet-hyp/end2end.onnx \
        --saveEngine=./work_dirs/yolox_rtmdet-hyp/end2end_trtexec_fp32.plan \
        --workspace=100000000 \
        --buildOnly \
        --fp16 \
        --verbose \