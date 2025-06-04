from typing import Dict, Optional

from .overlap import overlap_metrics
from .registration import registration_metrics

def compute_metrics(predictions: Dict, ground_truth: Dict, info: Optional[Dict] = None) -> Dict:
    metrics = {}
    
    if "overlap_pred" in predictions and "overlap_gt" in ground_truth:
        metrics.update(overlap_metrics(predictions, ground_truth, info))

    if "abs_pose_pred" in predictions:
        metrics.update(registration_metrics(predictions, ground_truth, info))

    return metrics