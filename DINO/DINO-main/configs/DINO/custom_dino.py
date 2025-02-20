_base_ = ['DINO_4scale.py']

# 修改为您数据集的类别数量
num_classes = 1  # 请替换为实际类别数

# 基础训练参数
lr = 2e-4  # 可以适当调小，比如 5e-5
lr_backbone = 1e-5
batch_size = 2  # 根据显存调整
weight_decay = 1e-4
epochs = 300    # 根据数据集大小调整
lr_drop = 40   # 在第40轮降低学习率

# 数据路径设置
data_path = '/storz/zkyang/DINO/DINO-main/images'
train_json = '/storz/zkyang/DINO/DINO-main/labels_coco/annotations.json'

# 数据增强设置
data_aug_scales = [480, 512, 544, 576, 608, 640]  # 可以根据您的图片尺寸调整
data_aug_max_size = 1333 # 根据您的图片尺寸调整
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]

# 模型基础设置
modelname = 'dino'
backbone = 'convnext_xlarge_22k'
backbone_dir = '/storz/zkyang/DINO/DINO-main/models/weights'
num_feature_levels = 4
hidden_dim = 256
num_queries = 500  # 可以根据您的数据集中最大目标数量调整

# DN设置优化
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
dn_labelbook_size = 1  # 正确，与num_classes相同

# 损失权重保持不变
cls_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

# 添加一些有用的配置
  # 指定输出目录
save_checkpoint_interval = 1  # 每轮都保存检查点
eval_interval = 1  # 每轮都进行评估

# 其他优化设置
use_ema = False  # 保持不变
aux_loss = True  # 启用自动混合精度训练 # 数据加载线程数