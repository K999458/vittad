import json
import os

def update_annotations(json_path, output_path):
    # 读取原始标注文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 为所有图片文件名添加.png扩展名
    for img in data['images']:
        if not img['file_name'].endswith('.png'):
            img['file_name'] = img['file_name'] + '.png'
    
    # 保存更新后的标注文件
    with open(output_path, 'w') as f:
        json.dump(data, f)

# 运行更新
update_annotations(
    '/storz/zkyang/DINO/DINO-main/tad/annotations/instances_val2017.json',
    '/storz/zkyang/DINO/DINO-main/tad/annotations/instances_val2017.json'
)