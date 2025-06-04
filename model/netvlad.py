from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import MODEL

class NetVLAD(nn.Module):
    def __init__(self, num_clusters: int = 64, dim: int = 32, alpha: float = 100.0, normalize_input: bool = True, intra_normalization: bool = True, l2_normalization: bool = True) -> None:
        super(NetVLAD, self).__init__()

        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.intra_normalization = intra_normalization
        self.l2_normalization = l2_normalization

        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1))
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()
    
    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(dim=-1).unsqueeze(dim=-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x: torch.Tensor):
        if self.normalize_input:
            x = F.normalize(x, dim=-1)
        
        x = x.transpose(-1, -2)
        soft_assign = self.conv(x.unsqueeze(dim=-2))
        soft_assign = F.softmax(soft_assign, dim=1)
        residual = x.unsqueeze(dim=1) - self.centroids[None, :, :, None]
        residual *= soft_assign

        vlad = torch.sum(residual, dim=-1)
        if self.intra_normalization:
            vlad = F.normalize(vlad, dim=-1)
        vlad = vlad.flatten(start_dim=1)  
        if self.l2_normalization:
            vlad = F.normalize(vlad, dim=-1)

        return vlad

@MODEL
class NetVLADPredictor(nn.Module):
    def __init__(self, drop_ratio: float, vlad_cfg: Dict, refine_k: int, beta: float, mode: str) -> None:
        super(NetVLADPredictor, self).__init__()

        self.drop = nn.Dropout(drop_ratio)
        self.pool = NetVLAD(**vlad_cfg)
        self.refine_k = refine_k
        self.beta = beta

        self.mode = mode
    
    def super_global(self, global_desc: torch.Tensor) -> torch.Tensor:
        k = min(self.refine_k + 1, global_desc.shape[0])
        
        dis = torch.cdist(global_desc, global_desc)
        neighbor_dis, neighbor_idx = torch.topk(dis, k=k, dim=1, largest=False)
        neighbor_desc = global_desc[neighbor_idx] # n * k * dim
        weight = torch.sum(neighbor_desc * global_desc.unsqueeze(dim=1), dim=-1, keepdim=True) # n * k * 1
        weight[:, 1:, :] *= self.beta

        refined_desc = torch.sum(weight * neighbor_desc, dim=1) / torch.sum(weight, dim=1)

        return refined_desc

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        desc_list = data_dict["descriptors"]
        desc_list = [self.pool(self.drop(desc)) for desc in desc_list]
        global_desc = torch.cat(desc_list, dim=0)

        if not self.training:
            global_desc = self.super_global(global_desc)

        if self.mode == "overlap":
            overlap = (2 - torch.cdist(global_desc, global_desc, p=2)) / 2.0
            overlap = overlap - torch.diag(torch.diag(overlap))

            prediction = {
                "overlap_pred": overlap
            }
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

        return prediction

    def create_input(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "descriptors": data_batch["descriptors"]
        }
    
    def create_ground_truth(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "overlap_gt": data_batch["gt_overlap"].squeeze_(dim=0)
        }