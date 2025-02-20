_base_ = ['DINO/DINO_4scaled.py']

# 修改基础配置中的一些值
num_classes = 2
modelname = 'dino'
backbone = 'resnet50'

# 数据集配置 - 直接在顶层定义，而不是在 data 字典中

#h5_path = '/store/zkyang/yolo_output/chrom_images_20241119_132716.h5'
#annotation_path = '/storz/zkyang/DINO/DINO-main/tad/all_tad_pixel_coords.csv'

# DN 相关配置
dn_labelbook_size = 2

# 训练相关配置
epochs = 12
lr = 2e-4
batch_size = 4
weight_decay = 1e-4

# 在现有配置的基础上添加：
visualize_eval = True  # 默认不开启可视化
vis_threshold = 0.3  # 可视化的置信度阈值
