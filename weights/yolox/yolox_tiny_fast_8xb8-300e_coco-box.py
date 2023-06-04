default_scope = 'mmyolo'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=30,
        save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_8xb8-300e_coco/yolox_tiny_8xb8-300e_coco_20220919_090908-0e40a6fc.pth'
resume = False
file_client_args = dict(backend='disk')
_file_client_args = dict(backend='disk')
tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300))
img_scales = [(640, 640), (320, 320), (960, 960)]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'mmdet.Resize',
            'scale': (416, 416),
            'keep_ratio': True
        }, {
            'type': 'mmdet.Resize',
            'scale': (320, 320),
            'keep_ratio': True
        }, {
            'type': 'mmdet.Resize',
            'scale': (640, 640),
            'keep_ratio': True
        }],
                    [{
                        'type': 'mmdet.RandomFlip',
                        'prob': 1.0
                    }, {
                        'type': 'mmdet.RandomFlip',
                        'prob': 0.0
                    }],
                    [{
                        'type': 'mmdet.Pad',
                        'pad_to_square': True,
                        'pad_val': {
                            'img': (114.0, 114.0, 114.0)
                        }
                    }],
                    [{
                        'type':
                        'mmdet.PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction')
                    }]])
]
data_root = './data/colorful-dataset/'
train_ann_file = 'annotations/instances_train2017.json'
train_data_prefix = 'train2017/'
val_ann_file = 'annotations/instances_val2017.json'
val_data_prefix = 'val2017/'
num_classes = 1
train_batch_size_per_gpu = 8
train_num_workers = 8
persistent_workers = True
base_lr = 0.01
max_epochs = 300
model_test_cfg = dict(
    yolox_style=True,
    multi_label=True,
    score_thr=0.001,
    max_per_img=300,
    nms=dict(type='nms', iou_threshold=0.65))
img_scale = (640, 640)
dataset_type = 'YOLOv5CocoDataset'
val_batch_size_per_gpu = 1
val_num_workers = 8
deepen_factor = 0.33
widen_factor = 0.375
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
batch_augments_interval = 10
weight_decay = 0.0005
loss_cls_weight = 1.0
loss_bbox_weight = 5.0
loss_obj_weight = 1.0
loss_bbox_aux_weight = 1.0
center_radius = 2.5
num_last_epochs = 15
random_affine_scaling_ratio_range = (0.1, 2)
mixup_ratio_range = (0.8, 1.6)
save_epoch_intervals = 10
max_keep_ckpts = 3
ema_momentum = 0.0001
model = dict(
    type='YOLODetector',
    init_cfg=dict(
        type='Kaiming',
        layer='Conv2d',
        a=2.23606797749979,
        distribution='uniform',
        mode='fan_in',
        nonlinearity='leaky_relu'),
    use_syncbn=False,
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='YOLOXBatchSyncRandomResize',
                random_size_range=(320, 640),
                size_divisor=32,
                interval=10)
        ]),
    backbone=dict(
        type='YOLOXCSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.375,
        out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOXPAFPN',
        deepen_factor=0.33,
        widen_factor=0.375,
        in_channels=[256, 512, 1024],
        out_channels=256,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOXHead',
        head_module=dict(
            type='YOLOXHeadModule',
            num_classes=1,
            in_channels=256,
            feat_channels=256,
            widen_factor=0.375,
            stacked_convs=2,
            featmap_strides=(8, 16, 32),
            use_depthwise=False,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox_aux=dict(
            type='mmdet.L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.SimOTAAssigner',
            center_radius=2.5,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'))),
    test_cfg=dict(
        yolox_style=True,
        multi_label=True,
        score_thr=0.001,
        max_per_img=300,
        nms=dict(type='nms', iou_threshold=0.65)))
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True)
]
train_pipeline_stage1 = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-320, -320)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='mmdet.Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='mmdet.PackDetInputs')
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='./data/colorful-dataset/',
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Mosaic',
                img_scale=(640, 640),
                pad_val=114.0,
                pre_transform=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk')),
                    dict(type='LoadAnnotations', with_bbox=True)
                ]),
            dict(
                type='mmdet.RandomAffine',
                scaling_ratio_range=(0.5, 1.5),
                border=(-320, -320)),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'flip', 'flip_direction'))
        ],
        metainfo=dict(classes=('box', ), palette=[(20, 220, 60)])))
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='mmdet.Resize', scale=(416, 416), keep_ratio=True),
    dict(
        type='mmdet.Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='./data/colorful-dataset/',
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='mmdet.Resize', scale=(416, 416), keep_ratio=True),
            dict(
                type='mmdet.Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        metainfo=dict(classes=('box', ), palette=[(20, 220, 60)])))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='./data/colorful-dataset/',
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='mmdet.Resize', scale=(416, 416), keep_ratio=True),
            dict(
                type='mmdet.Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        metainfo=dict(classes=('box', ), palette=[(20, 220, 60)])))
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='./data/colorful-dataset/val.json',
    metric='bbox')
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='./data/colorful-dataset/val.json',
    metric='bbox')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
param_scheduler = [
    dict(
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0005,
        begin=5,
        end=285,
        T_max=285,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=285, end=300)
]
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=15,
        new_train_pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='mmdet.Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='mmdet.PackDetInputs')
        ],
        priority=48),
    dict(type='mmdet.SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    val_interval=10,
    dynamic_intervals=[(285, 1)])
auto_scale_lr = dict(base_batch_size=64)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
scaling_ratio_range = (0.5, 1.5)
test_img_scale = (416, 416)
tta_img_scales = [(416, 416), (320, 320), (640, 640)]
class_name = ('box', )
metainfo = dict(classes=('box', ), palette=[(20, 220, 60)])
image_scale = (416, 416)
launcher = 'pytorch'
work_dir = './work_dirs/yolox_tiny_fast_8xb8-300e_coco-box'
