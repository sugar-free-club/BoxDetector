./tao-converter -k OW1paDZ2Zm1zaHNlM2ljbmZjdml0MDh2OHY6YzAyNGY2ZGMtNGQ3OS00NmI4LTg4YTItY2ViODM5N2EwMDIw \
                   -d 3,300,300 \
                   -o NMS \
                   -e ./TRT_ssd_resnet18.bin \
                   -m 1 \
                   -t fp32 \
                   -i nchw \
                   ./ssd_resnet18.etlt

./tao-converter -k OW1paDZ2Zm1zaHNlM2ljbmZjdml0MDh2OHY6YzAyNGY2ZGMtNGQ3OS00NmI4LTg4YTItY2ViODM5N2EwMDIw \
                   -d 3,300,300 \
                   -o NMS \
                   -e ./TRT_ssd_resnet18_fp16.bin \
                   -m 1 \
                   -t fp16 \
                   -i nchw \
                   ./ssd_resnet18.etlt