from typing import Union, Tuple

import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import open3d as o3d

import knn_search

from .process import transform, integrate_trans

def nearest_search(ref: Union[torch.Tensor, npt.NDArray], query: Union[torch.Tensor, npt.NDArray]):
    """
    :param ref: (dim, num)
    :param query: (dim, num)    
    """

    d, i = knn_search.knn_search(ref, query, 1)
    i -= 1
    return d, i

def mutual_match(descriptor0, descriptor1):
    """
    descriptor0: (num, dim)
    descriptor1: (num, dim)
    """
    descriptor0, descriptor1 = descriptor0.T.contiguous(), descriptor1.T.contiguous()

    d0, i0 = nearest_search(descriptor1, descriptor0)
    d1, i1 = nearest_search(descriptor0, descriptor1)

    i0 = i0[0]
    i1 = i1[0]

    d0 = d0[0]

    match_idx = torch.where(i1[i0] == torch.arange(len(i0), device=i0.device))[0]
    match = torch.stack([torch.arange(len(i0), device=i0.device), i0], dim=-1)
    match = match[match_idx]
    dis = d0[match_idx]

    return match, dis

def ransac(pc0: npt.NDArray, pc1: npt.NDArray, match: npt.NDArray, iteration: int = 50000, max_correspondence_distance: float = 0.07):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(pc0)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(pc1)
    coores = o3d.utility.Vector2iVector(match.cpu().numpy())

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source_pcd, target_pcd, coores, max_correspondence_distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        o3d.pipelines.registration.RANSACConvergenceCriteria(iteration, 1000)
    )

    trans = result.transformation
    trans = np.linalg.inv(trans)

    return trans

def svd(pc0: torch.tensor, pc1: torch.tensor):
    centroid0 = torch.mean(pc0, dim=0, keepdim=True)
    centroid1 = torch.mean(pc1, dim=0, keepdim=True)

    pc0_centered = pc0 - centroid0
    pc1_centered = pc1 - centroid1

    H = torch.matmul(pc0_centered.T, pc1_centered)
    U, _, V = torch.linalg.svd(H)
    R = torch.matmul(U, V)
    
    if torch.det(R) < 0:
        R[[0, 1]] = R[[1, 0]]

    t = centroid0 - torch.matmul(centroid1, R.T)

    trans = integrate_trans(R, t.T)

    return trans

def inlier_ratio(kpc0, kpc1, trans, max_correspondence_distance: float = 0.07):
    kpc1_t = transform(kpc1, trans)
    dis = torch.sum(torch.square(kpc0 - kpc1_t), dim=-1)

    inlier_index = torch.where(dis < max_correspondence_distance ** 2)[0]
    ir = inlier_index.shape[0] #/ dis.shape[0]

    return ir

def refine(kpc0, kpc1, trans, max_correspondence_distance: float = 0.07):
    kpc1_t = transform(kpc1, trans)
    
    diff = torch.sum(torch.square(kpc0 - kpc1_t), dim=-1)
    overlap = torch.where(diff < max_correspondence_distance ** 2)[0]

    if overlap.shape[0] > 0:
        kpc0 = kpc0[overlap]
        kpc1 = kpc1[overlap]

        trans = svd(kpc0, kpc1)

    return trans

def pairwise_registration_cuda(
        pc0: npt.NDArray,
        pc1: npt.NDArray,
        pc0_ts: torch.tensor,
        pc1_ts: torch.tensor,
        descriptor0: torch.tensor,
        descriptor1: torch.tensor,
        max_correspondence_distance: float = 0.07
    ) -> Tuple[torch.tensor, float, float]:

    
    match, dis = mutual_match(descriptor0, descriptor1)
   
    trans = ransac(pc0, pc1, match)
    trans = torch.from_numpy(trans).float().cuda()

    kpc0 = pc0_ts[match[:, 0]]
    kpc1 = pc1_ts[match[:, 1]]

    trans = refine(kpc0, kpc1, trans, max_correspondence_distance * 2)
    trans = refine(kpc0, kpc1, trans, max_correspondence_distance)

    inlier_info = inlier_ratio(kpc0, kpc1, trans, max_correspondence_distance)

    return trans, inlier_info, inlier_info