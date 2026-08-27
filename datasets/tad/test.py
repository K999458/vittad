import sys
import os
import argparse
import torch

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

from tad.tad import build as build_tad

def get_args():
    parser = argparse.ArgumentParser(description="Test TAD Dataset Loader")
    parser.add_argument('--coco_path', type=str, required=True, help='Path to image directory')
    return parser.parse_args()

def main():
    args = get_args()
    print(f"\n=== 参数信息 ===")
    print(f"图片目录: {args.coco_path}")
    print("===============\n")
    
    try:
        dataset = build_tad(image_set='val', args=args)
        print(f"\n数据集包含 {len(dataset)} 张图片")
        
        # 测试加载前5张图片
        for i in range(min(5, len(dataset))):
            try:
                img, target = dataset[i]
                print(f"图片 {i} 加载成功:")
                print(f"  文件名: {target['file_name']}")
                print(f"  尺寸: {target['orig_size'].tolist()}")
            except Exception as e:
                print(f"加载图片失败: {e}")
                
    except Exception as e:
        print(f"构建数据集时出错: {e}")
        raise

if __name__ == '__main__':
    main()