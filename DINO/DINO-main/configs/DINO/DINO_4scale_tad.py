from .DINO_4scale_swin import model, optimizer, lr_config, train_pipeline, test_pipeline

# 修改数据配置
data = dict(
    samples_per_gpu=2,  # batch size
    workers_per_gpu=2,  # 数据加载线程数
    train=dict(
        type='CocoDataset',
        ann_file='/storz/zkyang/DINO/DINO-main/tad/train_annotations.json',
        img_prefix='',  # 如果图片是在h5文件中，这里留空
        pipeline=train_pipeline
    ),
    val=dict(
        type='CocoDataset',
        ann_file='/storz/zkyang/DINO/DINO-main/tad/val_annotations.json',
        img_prefix='',
        pipeline=test_pipeline
    ),
    test=dict(
        type='CocoDataset',
        ann_file='/storz/zkyang/DINO/DINO-main/tad/val_annotations.json',
        img_prefix='',
        pipeline=test_pipeline
    )
)

# 修改类别数量
model.update(dict(
    bbox_head=dict(
        num_classes=1  # 只有一个类别 "tad"
    )
))

# 设置训练参数
total_epochs = 12