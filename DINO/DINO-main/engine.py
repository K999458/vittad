# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device
import cv2
import numpy as np
import torch
from PIL import Image

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        samples = samples.to(device)
        
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat



def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        # 导入补丁
        from util.coco_patch import COCOeval
        import numpy as np
        
        # 添加可视化控制参数
        do_vis = getattr(args, 'visualize_eval', False)  # 从args中获取visualize_eval参数
        if do_vis:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            print(f"将保存可视化结果到: {vis_dir}")
        
        if base_ds is None and hasattr(data_loader.dataset, 'coco'):
            base_ds = data_loader.dataset.coco
            print("从数据集获取COCO API成功")
        
        if base_ds is None:
            print("警告：无法获取COCO API，将只进行推理而不评估性能")
            
        model.eval()
        criterion.eval()

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'

        final_res = []
        
        for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
            samples = samples.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]

            outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            
            # 收集检测结果
            for batch_idx, (image_id, outputs) in enumerate(res.items()):
                _scores = outputs['scores'].tolist()
                _labels = outputs['labels'].tolist()
                _boxes = outputs['boxes'].tolist()
                
                # 可视化部分
                if do_vis:
                    # 读取原始图片
                    img_name = targets[batch_idx]['file_name']
                    img_path = os.path.join(args.coco_path, 'images', img_name)
                    # 使用PIL读取图片以确保正确的颜色
                    orig_img = Image.open(img_path).convert('RGB')
                    # 转换为OpenCV格式
                    vis_img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
                    
                    # 获取原始图像尺寸
                    orig_h, orig_w = vis_img.shape[:2]
                    has_results = False
                    
                    # 绘制检测框和分数
                    for score, label, box in zip(_scores, _labels, _boxes):
                        if score > 0.3:  # 置信度阈值
                            has_results = True
                            # 获取框坐标
                            x1, y1, x2, y2 = map(int, box)
                            
                            # 确保坐标在图像范围内
                            x1 = max(0, min(orig_w-1, x1))
                            x2 = max(0, min(orig_w-1, x2))
                            y1 = max(0, min(orig_h-1, y1))
                            y2 = max(0, min(orig_h-1, y2))
                            
                            # 绘制边界框
                            cv2.rectangle(vis_img, 
                                        (x1, y1), 
                                        (x2, y2), 
                                        (0, 255, 0), 2)  # 绿色框
                            
                            # 添加标签和分数
                            label_text = f'Class {label}: {score:.2f}'
                            cv2.putText(vis_img, 
                                      label_text,
                                      (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      1, 
                                      (0, 255, 0),
                                      2)
                    
                    # 只保存有检测结果的图片
                    if has_results:
                        save_path = os.path.join(vis_dir, f'vis_{os.path.splitext(img_name)[0]}.jpg')
                        cv2.imwrite(save_path, vis_img)
                
                # 收集检测结果
                for s, l, b in zip(_scores, _labels, _boxes):
                    if s > 0.3:  # 置信度阈值
                        itemdict = {
                            "image_id": int(image_id),
                            "category_id": l,
                            "bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]],  # 转换为COCO格式 [x,y,w,h]
                            "score": s,
                        }
                        final_res.append(itemdict)

        # 保存预测结果
        if args.output_dir:
            import json
            with open(os.path.join(output_dir, f'results{args.rank}.json'), 'w') as f:
                json.dump(final_res, f)

        # 评估统计
        stats = {}
        if base_ds is not None:
            try:
                from pycocotools.coco import COCO
                
                # 如果base_ds已经是COCO对象，直接使用
                if isinstance(base_ds, COCO):
                    coco_gt = base_ds
                else:
                    # 否则创建新的COCO对象
                    coco_gt = COCO()
                    coco_gt.dataset = base_ds
                    coco_gt.createIndex()
                
                # 处理检测结果中的数值类型
                for item in final_res:
                    item['score'] = float(item['score'])
                    item['bbox'] = [float(x) for x in item['bbox']]
                
                # 加载检测结果
                coco_dt = coco_gt.loadRes(final_res) if final_res else None
                
                if coco_dt is not None:
                    # 使用修补后的COCOeval
                    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    stats = {'coco_eval_bbox': [float(x) for x in coco_eval.stats.tolist()]}
                else:
                    print("Warning: No valid detections for evaluation")
                    stats = {'coco_eval_bbox': [0.0] * 12}
                    
            except Exception as e:
                print(f"Warning: Error in COCO evaluation: {e}")
                import traceback
                traceback.print_exc()
                stats = {'coco_eval_bbox': [0.0] * 12}
        else:
            print("Warning: No ground truth annotations available for evaluation")
            stats = {'coco_eval_bbox': [0.0] * 12}

        return stats, None

    except Exception as e:
        print(f"评估过程发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, None

# 图像预处理的均值和标准差
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res



