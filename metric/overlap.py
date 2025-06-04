from typing import Dict

import torch
import numpy as np

def overlap_metrics(predictions: Dict, ground_truth: Dict, info: Dict) -> Dict:
    overlap_pred = predictions["overlap_pred"].detach()
    overlap_gt = ground_truth["overlap_gt"]

    if overlap_gt.sum() == 0:
        return {}

    k = int(min(info.get("k", 10), overlap_gt.shape[0]/2.0))
    frames = overlap_gt.shape[0]
    truely_pre = 0
    truely_gt  = k * frames
    selected_min, seleted_mean, seleted_median = 0, 0, 0
    
    for f in range(frames):
        gt = overlap_gt[f]
        pred = overlap_pred[f]
        # gt from large to small
        arg_gt = torch.argsort(gt, descending=True)[0:k]
        # pre from large to small
        arg_pre = torch.argsort(pred, descending=True)[0:k]
        for i in arg_pre:
            if i in arg_gt:
                truely_pre += 1
        
        seleted_overlap = gt[arg_pre].cpu().numpy()
        selected_min += np.min(seleted_overlap)
        seleted_mean += np.mean(seleted_overlap)
        seleted_median += np.median(seleted_overlap)

    recall = np.array(truely_pre / truely_gt)
    selected_min = np.array(selected_min / frames)
    seleted_mean = np.array(seleted_mean / frames)
    seleted_median = np.array(seleted_median / frames)

    metrics = {
        "topk_recall": recall,
        "selected_min": selected_min,
        "selected_mean": seleted_mean,
        "selected_median": seleted_median
    }
    
    return metrics