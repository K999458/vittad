# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from util.misc import all_gather


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, useCats=True):
        """
        初始化COCO评估器
        Args:
            coco_gt: COCO数据集对象或None
            iou_types: 评估类型列表，如['bbox']
            useCats: 是否使用类别信息
        """
        assert isinstance(iou_types, (list, tuple))
        
        # 检查coco_gt的类型
        if isinstance(coco_gt, torch.device):
            print("Error: coco_gt is a torch.device object")
            coco_gt = None
            
        if coco_gt is None:
            print("Warning: coco_gt is None, creating empty COCO object")
            coco_gt = COCO()
            # 初始化基本数据结构
            coco_gt.dataset = {
                'images': [],
                'annotations': [],
                'categories': [{'id': 0, 'name': 'target'}]
            }
            coco_gt.createIndex()
        
        # 验证COCO对象的有效性
        if not hasattr(coco_gt, 'dataset') or not isinstance(coco_gt.dataset, dict):
            raise ValueError("Invalid COCO ground truth object")
        
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        
        # 初始化每种评估类型
        for iou_type in iou_types:
            try:
                # 确保必要的数据结构存在
                if 'images' not in self.coco_gt.dataset:
                    self.coco_gt.dataset['images'] = []
                if 'annotations' not in self.coco_gt.dataset:
                    self.coco_gt.dataset['annotations'] = []
                if 'categories' not in self.coco_gt.dataset:
                    self.coco_gt.dataset['categories'] = [{'id': 0, 'name': 'target'}]
                
                # 重新创建索引
                if not hasattr(self.coco_gt, 'imgs') or not self.coco_gt.imgs:
                    self.coco_gt.createIndex()
                
                # 初始化评估器
                self.coco_eval[iou_type] = COCOeval(self.coco_gt, iouType=iou_type)
                self.coco_eval[iou_type].params.useCats = useCats
                
                print(f"Successfully initialized {iou_type} evaluator")
                print(f"Number of images: {len(self.coco_gt.imgs)}")
                print(f"Number of annotations: {len(self.coco_gt.anns)}")
                print(f"Number of categories: {len(self.coco_gt.cats)}")
                
            except Exception as e:
                print(f"Error initializing COCOeval for {iou_type}: {str(e)}")
                print(f"COCO GT structure: {self.coco_gt.dataset.keys() if hasattr(self.coco_gt, 'dataset') else 'No dataset attribute'}")
                raise e

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        """
        更新评估器的预测结果
        """
        if not predictions:
            print("Warning: Empty predictions")
            return
            
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            try:
                results = self.prepare(predictions, iou_type)
                if not results:
                    print(f"Warning: No results for {iou_type}")
                    continue
                
                # 打印一些调试信息
                print(f"Processing {len(results)} results for {iou_type}")
                if results:
                    print(f"Sample result: {results[0]}")
                
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
                coco_eval = self.coco_eval[iou_type]
                coco_eval.cocoDt = coco_dt
                coco_eval.params.imgIds = list(img_ids)
                coco_eval._prepare()
                
            except Exception as e:
                print(f"Warning: Error in COCO evaluation for {iou_type}: {str(e)}")
                print(f"Number of results: {len(results) if results else 0}")
                continue

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            if not isinstance(prediction["scores"], list):
                scores = prediction["scores"].tolist()
            else:
                scores = prediction["scores"]
            if not isinstance(prediction["labels"], list):
                labels = prediction["labels"].tolist()
            else:
                labels = prediction["labels"]

        
            try:
                coco_results.extend(
                    [
                        {
                            "image_id": original_id,
                            "category_id": labels[k],
                            "bbox": box,
                            "score": scores[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
            except:
                import ipdb; ipdb.set_trace()
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)

    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
