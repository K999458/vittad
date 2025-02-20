import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class HiCTADDataset(Dataset):
    def __init__(self, h5_path, annotation_path, image_set='train'):
        """
        Args:
            h5_path: HiC矩阵的h5文件路径
            annotation_path: TAD标注文件路径
            image_set: 'train' 或 'val'
        """
        self.h5_file = h5py.File(h5_path, 'r')
        self.transform = make_transforms(image_set)
        
        # 加载并处理标注
        self.annotations = pd.read_csv(annotation_path)
        self.grouped_annotations = self._process_annotations()
        self.image_names = sorted(list(self.grouped_annotations.keys()))
        
    def _process_annotations(self):
        """处理标注文件，将每张图的TAD标注组织在一起"""
        grouped = {}
        for _, row in self.annotations.iterrows():
            image_name = row['image_name']
            coords = eval(row['pixel_coords'])  # 转换字符串坐标为元组
            
            if image_name not in grouped:
                grouped[image_name] = []
            grouped[image_name].append(coords)
            
        return grouped
    
    def _get_triangle_points(self, point):
        """从对角线上的点生成三角形的三个顶点"""
        x, y = point
        return [(x,x), (y,x), (y,y)]
    
    def _convert_triangle_to_box(self, triangle_points):
        """将TAD三角形转换为边界框"""
        points = np.array(triangle_points)
        x_min = min(points[:, 0])
        y_min = min(points[:, 1])
        x_max = max(points[:, 0])
        y_max = max(points[:, 1])
        return [x_min, y_min, x_max, y_max]
    
    def __getitem__(self, idx):
        # 获取图像名称和数据
        image_name = self.image_names[idx]
        chrom = f"chr{image_name.split('_')[1]}"  # chr_10_1 -> chr10
        num = int(image_name.split('_')[2]) - 1   # chr_10_1 -> 0
        
        # 获取图像数据
        image = self.h5_file[f"{chrom}/images"][num]
        image = np.nan_to_num(image, nan=0.0)
        image = torch.from_numpy(image).float()
        
        # 获取该图像的所有TAD标注
        tad_points = self.grouped_annotations[image_name]
        boxes = []
        for point in tad_points:
            triangle = self._get_triangle_points(point)
            box = self._convert_triangle_to_box(triangle)
            boxes.append(box)
            
        # 转换为tensor格式
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # 构建目标字典
        target = {
            'boxes': boxes,
            'labels': torch.ones((len(boxes),), dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        if self.transform:
            image = self.transform(image)
            
        return image, target
    
    def __len__(self):
        return len(self.image_names)

def make_transforms(image_set):
    """
    构建数据转换管道
    """
    transforms = []
    
    # 添加通道维度
    transforms.append(T.Lambda(lambda x: x.unsqueeze(0).repeat(3, 1, 1)))
    
    if image_set == 'train':
        transforms.extend([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ])
    
    # 标准化
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]))
    
    return T.Compose(transforms)