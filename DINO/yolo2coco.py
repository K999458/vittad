import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def visualize_annotations(image_path, annotations, output_path, num_samples=3):
    """可视化标注并保存图片"""
    # 创建可视化输出目录
    vis_output_path = os.path.join(output_path, 'visualization')
    os.makedirs(vis_output_path, exist_ok=True)
    
    # 获取要可视化的图片
    image_ids = sorted(list(set([ann['image_id'] for ann in annotations['annotations']])))
    image_ids = image_ids[:num_samples]  # 只取前num_samples张图片
    
    for image_id in image_ids:
        # 获取图片信息
        img_info = next(img for img in annotations['images'] if img['id'] == image_id)
        img_file = img_info['file_name']
        
        # 读取图片
        img = cv2.imread(os.path.join(image_path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建新的图形
        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        
        # 绘制该图片的所有标注框
        current_anns = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
        for ann in current_anns:
            x, y, w, h = ann['bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2)
            plt.gca().add_patch(rect)
            
            # 添加标签
            plt.text(x, y, f'ID: {ann["category_id"]}', 
                    color='white', backgroundcolor='red', fontsize=8)
        
        plt.title(f'Image: {img_file}\nTotal annotations: {len(current_anns)}')
        plt.axis('off')
        
        # 保存图片
        output_file = os.path.join(vis_output_path, f'vis_{img_file}')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"已保存可视化结果到: {output_file}")

def yolo_to_coco():
    # 基础路径
    base_path = '/storz/zkyang/DINO/DINO-main'
    yolo_label_path = os.path.join(base_path, 'labels')
    image_path = os.path.join(base_path, 'images')
    output_path = os.path.join(base_path, 'labels_coco')
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 初始化COCO格式
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "target"}]
    }
    
    annotation_id = 0
    
    # 遍历所有图片和标签
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith(('.jpg', '.png'))])
    
    print(f"开始处理 {len(image_files)} 个文件...")
    
    for image_id, image_file in enumerate(tqdm(image_files)):
        # 读取图片获取尺寸
        img = cv2.imread(os.path.join(image_path, image_file))
        if img is None:
            print(f"警告：无法读取图片 {image_file}")
            continue
            
        height, width = img.shape[:2]
        
        # 添加图片信息
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_file,
            "height": height,
            "width": width
        })
        
        # 读取对应的标签文件
        label_file = os.path.join(yolo_label_path, image_file.rsplit('.', 1)[0] + '.txt')
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    
                    # YOLO坐标转换为COCO坐标
                    x = (x_center - w/2) * width
                    y = (y_center - h/2) * height
                    w = w * width
                    h = h * height
                    
                    # 添加标注信息
                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    annotation_id += 1
    
    # 保存COCO格式的JSON文件
    json_output_path = os.path.join(output_path, 'annotations.json')
    with open(json_output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"\nCOCO格式转换完成！")
    print(f"- 处理图片数量: {len(coco_format['images'])}")
    print(f"- 处理标注数量: {len(coco_format['annotations'])}")
    print(f"- JSON文件保存在: {json_output_path}")
    
    # 进行可视化
    print("\n开始生成可视化结果...")
    visualize_annotations(image_path, coco_format, output_path)

if __name__ == "__main__":
    yolo_to_coco()