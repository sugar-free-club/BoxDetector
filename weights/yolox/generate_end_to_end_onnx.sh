python3.7 projects/easydeploy/tools/export.py \
    ./work_dirs/yolox/yolox_tiny_fast_8xb8-300e_coco-box.py \
    ./work_dirs/yolox/best_coco_bbox_mAP_epoch_300.pth \
    --work-dir work_dirs/yolox \
    --img-size 416 416 \
    --batch 1 \
    --device cuda:0 \
    --simplify \
    --opset 11 \
    --backend 2 \
    --pre-topk 1000 \
    --keep-topk 100 \
    --iou-threshold 0.65 \
    --score-threshold 0.25