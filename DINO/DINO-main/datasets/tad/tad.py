import os
import torch
from PIL import Image
import datasets.transforms as T

def make_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize([400, 500, 600], max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize([800], max_size=1333),
                ])
            ),
            normalize,
        ])
    elif image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

class TADDetection(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms, return_masks=False):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks
        
        # 加载COCO标注
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # 获取类别信息
        self.categories = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}
        
        # 记录模式
        self.is_train = 'RandomHorizontalFlip' in str(transforms)
        mode = "训练" if self.is_train else "验证"
        print(f"加载了 {len(self.ids)} 张{mode}图片")
        print(f"包含 {len(self.categories)} 个类别")

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # 加载图片
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # 加载标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h] 格式
            # 转换为 [x1, y1, x2, y2] 格式
            bbox = [
                float(bbox[0]),  # x1
                float(bbox[1]),  # y1
                float(bbox[0] + bbox[2]),  # x2 = x1 + w
                float(bbox[1] + bbox[3])   # y2 = y1 + h
            ]
            boxes.append(bbox)
            labels.append(ann['category_id'])
            areas.append(float(ann['area']))
            iscrowd.append(ann['iscrowd'])
        
        # 确保所有数据都转换为tensor
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64),
            'orig_size': torch.as_tensor([int(h), int(w)], dtype=torch.int64),
            'size': torch.as_tensor([int(h), int(w)], dtype=torch.int64),
            'file_name': img_info['file_name']  # 这个保持字符串格式
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

def build(image_set, args):
    root = args.coco_path
    img_folder = os.path.join(root, "images")
    ann_folder = os.path.join(root, "annotations")
    
    if image_set == 'train':
        ann_file = os.path.join(ann_folder, "annotations.json")
        print(f"加载训练数据集:")
    elif image_set == 'val':
        ann_file = os.path.join(ann_folder, "annotations.json")
        print(f"加载验证数据集:")
    else:
        raise ValueError(f'未知的数据集类型: {image_set}')
        
    print(f"图片文件夹: {img_folder}")
    print(f"标注文件: {ann_file}")

    assert os.path.exists(img_folder), f'图片路径不存在: {img_folder}'
    assert os.path.exists(ann_file), f'标注文件不存在: {ann_file}'

    dataset = TADDetection(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=make_transforms(image_set),
        return_masks=args.masks if hasattr(args, 'masks') else False
    )

    return dataset