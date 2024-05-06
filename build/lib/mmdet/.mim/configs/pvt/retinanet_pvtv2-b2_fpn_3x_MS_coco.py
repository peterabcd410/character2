_base_ = 'retinanet_pvtv2-b0_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b2.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))

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

train_dataloader = dict(batch_size=2, dataset=dict(pipeline=train_pipeline))

max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)