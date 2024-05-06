_base_ = [
    "../_base_/models/retinanet_r50_fpn.py",
    # "../_base_/datasets/coco_detection.py",
    "../_base_/datasets/coco_detection_six.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

dataset_type = "CocoDataset_six"
classes = (
    "container ship",
    "liner",
    "other ship",
    "bulk carrier",
    "sailboat",
    "island reef",
)

pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"  # noqa

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    bbox_head=dict(
        type="YOLOXHead",
        num_classes=6,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
        loss_cls=dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0
        ),
        loss_bbox=dict(
            type="IoULoss", mode="square", eps=1e-16, reduction="sum", loss_weight=5.0
        ),
        loss_obj=dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0
        ),
        loss_l1=dict(type="L1Loss", reduction="sum", loss_weight=1.0),
    ),
    # training and testing settings
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
)

# augmentation stratgy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args={{_base_.backend_args}}),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(batch_size=4, dataset=dict(pipeline=train_pipeline))

max_epochs = 100
train_cfg = dict(max_epochs=max_epochs)

test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))
# learning rate
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1,
    ),
]

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
    optimizer=dict(
        _delete_=True, type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05
    ),
)
