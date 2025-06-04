from typing import Dict

import os
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import open3d as o3d
from tqdm import tqdm

from utils import pairwise_registration_cuda, mutual_match, svd, transform, inv_transform, integrate_trans

from .builder import MODEL, build_model


class Frame:
    def __init__(self, id, keypoints, descriptors):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id = id
        self.pose = torch.eye(4, device=device).float()
        self.keypoints = keypoints
        self.descriptors = descriptors

        self.descriptors_ts = torch.from_numpy(self.descriptors).float().to(device)
        self.keypoints_ts = torch.from_numpy(self.keypoints).float().to(device)

        self.redundancy = torch.ones((keypoints.shape[0], 1), dtype=torch.int8, device=device)

    def update(self, item, value):
        setattr(self, item, value)

@MODEL
class IMVR:    
    def __init__(self, descriptor: str, overlap_cfg: Dict, sample: int, k: int, ird: float, merge_method: str, ckpt_path: str):
        super().__init__()

        self.descriptor = descriptor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.overlap_predictor = build_model(overlap_cfg)
        
        model_weight = torch.load(ckpt_path, map_location="cpu")
        self.overlap_predictor.load_state_dict(model_weight)
        self.overlap_predictor.to(self.device).eval()

        self.sample = sample
        self.k = k
        self.ird = ird
        self.iteration = 10

        self.merge_method = merge_method

    def _load_pc(self, scene_path: str, index: int) -> npt.NDArray:
        pc_path = os.path.join(scene_path, "PointCloud", f"cloud_bin_{index}.ply")
        pc = o3d.io.read_point_cloud(pc_path)
        pc = np.asarray(pc.points).astype(np.float32)
        return pc

    def _load_keypoints_index(self, scene_path: str, index: int) -> npt.NDArray:
        assert self.descriptor == "yoho", "Only YOHO descriptor has index."
        key_index_path = os.path.join(scene_path, "Keypoints", f"cloud_bin_{index}Keypoints.txt")
        keypoints_index = np.loadtxt(key_index_path).astype(np.int64)
        return keypoints_index

    def _load_keypoints(self, scene_path: str, index: int) -> npt.NDArray:
        keypoints_path = os.path.join(scene_path, "Keypoints_PC_" + self.descriptor, f"cloud_bin_{index}Keypoints.npy")
        if not os.path.exists(keypoints_path):
            pc = self._load_pc(scene_path, index)
            keypoints_index = self._load_keypoints_index(scene_path, index)
            keypoints = pc[keypoints_index]
            if not os.path.exists(os.path.join(scene_path, "Keypoints_PC_" + self.descriptor)):
                os.makedirs(os.path.join(scene_path, "Keypoints_PC_" + self.descriptor))
            np.save(keypoints_path, keypoints)
        else:
            keypoints = np.load(keypoints_path).astype(np.float32)
        return keypoints
    
    def _load_descriptor(self, scene_path: str, index: int) -> npt.NDArray:
        descriptor_path = os.path.join(scene_path, self.descriptor + "_desc", f"{index}.npy")
        descriptor = np.load(descriptor_path).astype(np.float32)
        return descriptor
    
    def load_data(self, scene_path: str):
        self.frames = []

        self.total_frame = len(os.listdir(os.path.join(scene_path, self.descriptor + "_desc")))

        for i in range(self.total_frame):
            keypoints = self._load_keypoints(scene_path, i)
            descriptors = self._load_descriptor(scene_path, i)

            if self.sample < 5000:
                sample = np.random.choice(5000, self.sample, replace=False)
                keypoints = keypoints[sample]
                descriptors = descriptors[sample]

            self.frames.append(Frame(i, keypoints, descriptors))

        self.registration_overlap_ratio = torch.zeros(self.total_frame, self.total_frame, device=self.device)
        self.relative_transform = torch.eye(4, device=self.device).unsqueeze_(dim=0).unsqueeze_(dim=0).repeat(self.total_frame, self.total_frame, 1, 1)

    def get_overlap(self):        
        
        if self.overlap_predictor.__class__.__name__ == "NetVLADPredictor":
            data = {
                "descriptors": [self.frames[i].descriptors_ts.unsqueeze(dim=0) for i in range(self.total_frame)]
            } 
        else:
            raise NotImplementedError("Overlap predictor not implemented.")

        with torch.no_grad():
            overlap_out = self.overlap_predictor(data)
        overlap_mat = overlap_out["overlap_pred"]

        return overlap_mat

    def merge(self, base_frame: Frame, selected_index: int):
        frame_kp = self.frames[selected_index].keypoints_ts
        frame_desc = self.frames[selected_index].descriptors_ts
        frame_redundancy = self.frames[selected_index].redundancy
        base_kp = base_frame.keypoints_ts
        base_desc = base_frame.descriptors_ts
        base_redundancy = base_frame.redundancy

        match, dist = mutual_match(base_kp, frame_kp)
        match = match[dist < self.ird]

        base_mask = torch.zeros(base_kp.shape[0], dtype=torch.bool, device=self.device)
        base_mask[match[:, 0]] = True
        frame_mask = torch.zeros(frame_kp.shape[0], dtype=torch.bool, device=self.device)
        frame_mask[match[:, 1]] = True

        if self.merge_method == "sample":
            prob = torch.rand(match.shape[0], device=self.device).reshape(-1, 1)
            overlap_redundancy = base_redundancy[base_mask] + 1
            threshold = 1 / overlap_redundancy
            overlap_kp = torch.where(prob > threshold, base_kp[match[:, 0]], frame_kp[match[:, 1]])
            overlap_desc = torch.where(prob > threshold, base_desc[match[:, 0]], frame_desc[match[:, 1]])
        elif self.merge_method == "mean":
            overlap_kp = (base_kp[match[:, 0]] + frame_kp[match[:, 1]]) / 2
            overlap_desc = (base_desc[match[:, 0]] + frame_desc[match[:, 1]]) / 2
        elif self.merge_method == "base":
            overlap_kp = base_kp[match[:, 0]]
            overlap_desc = base_desc[match[:, 0]]
        elif self.merge_method == "local":
            overlap_kp = frame_kp[match[:, 1]]
            overlap_desc = frame_desc[match[:, 1]]
        elif self.merge_method == "concatenate":
            overlap_kp = torch.cat([base_kp[match[:, 0]], frame_kp[match[:, 1]]], dim=0)
            overlap_desc = torch.cat([base_desc[match[:, 0]], frame_desc[match[:, 1]]], dim=0)
        else:
            raise NotImplementedError("Merge method not implemented.")

        base_kp = torch.cat([base_kp[~base_mask], frame_kp[~frame_mask], overlap_kp], dim=0)
        base_desc = torch.cat([base_desc[~base_mask], frame_desc[~frame_mask], overlap_desc], dim=0)

        base_frame.update("keypoints_ts", base_kp)
        base_frame.update("descriptors_ts", base_desc)
        base_frame.update("keypoints", base_kp.cpu().numpy())

        if self.merge_method == "sample":
            base_redundancy = torch.cat([base_redundancy[~base_mask], frame_redundancy[~frame_mask], overlap_redundancy], dim=0)
            base_frame.update("redundancy", base_redundancy)

    def single_rotation_averaging(self, R: torch.Tensor, weights: torch.Tensor, R_init: torch.Tensor):
        r = R_init.reshape(1, 9)

        for i in range(self.iteration):
            v = R.flatten(start_dim=1) - r
            d = torch.norm(v, dim=-1, keepdim=True)

            r_prev = r

            r = torch.sum(weights * v / d, dim=0) / torch.sum(weights / d, dim=0)

            if torch.norm(r - r_prev) < 1e-3:
                break
        
        r = r.reshape(3, 3)
        U, _, V = torch.svd(r)
        r = torch.matmul(U, V.T)
        if torch.det(r) < 0:
            r[0:2] = r[[1, 0]]
        
        return r
    
    def transformation_averaging(self, trans: torch.Tensor, weights: torch.Tensor, trans_init: torch.Tensor):
        r = self.single_rotation_averaging(trans[:, :3, :3], weights, trans_init[:3, :3])

        B = trans[:, :3, 3:].reshape(-1, 1)
        A = torch.matmul(trans[:, :3, :3], r.T).reshape(-1, 3)
        W = torch.diag(weights.repeat(1, 3).flatten())

        t = torch.linalg.lstsq(A.T @ W @ A, A.T @ W @ B).solution.reshape(3, 1)
        trans = integrate_trans(r, t)

        return trans
    
    def local_refine(self, index: int, trans: torch.Tensor):
        kp_t = transform(self.frames[index].keypoints_ts, trans)
        
        trans_list = []
        weight_list = []
        for i in self.base_list:
            
            dist = torch.cdist(self.frames[i].keypoints_ts, kp_t)

            d0, i0 = torch.min(dist, dim=-1)
            d1, i1 = torch.min(dist, dim=-2)

            ror = (torch.sum(d0 < self.ird) + torch.sum(d1 < self.ird)) / (len(d0) + len(d1))
            self.registration_overlap_ratio[i, index] = self.registration_overlap_ratio[index, i] = ror

            if ror > 0.3:
                kp0 = self.frames[i].keypoints_ts[torch.arange(len(d0), device=d0.device)[d0 < self.ird]]
                kp1 = self.frames[index].keypoints_ts[i0[d0 < self.ird]]
                pairwise_trans = svd(kp0, kp1)
                trans_list.append(pairwise_trans)
                weight_list.append(ror)
                self.relative_transform[i, index] = pairwise_trans
                self.relative_transform[index, i] = inv_transform(pairwise_trans)

        if len(trans_list) > 1:
            trans = self.transformation_averaging(torch.stack(trans_list), torch.stack(weight_list).reshape(-1, 1), trans)
            kp_t = transform(self.frames[index].keypoints_ts, trans)

        self.base_list.append(index)
        self.frames[index].update("keypoints_ts", kp_t)
        self.frames[index].update("keypoints", kp_t.cpu().numpy())
        self.frames[index].update("pose", inv_transform(trans))    

    def registration(self, overlap: torch.Tensor):
    
        pivot = torch.argmax(torch.sum(overlap, dim=-1))
        self.base_list = [pivot]

        base_overlap = overlap[pivot]
        base_frame = self.frames[pivot]
    
        base_mask = torch.ones_like(base_overlap)
        base_mask[pivot] = 0
  
        for i in tqdm(range(self.total_frame - 1)):

            k = min(self.k, self.total_frame - i - 1)
            candidates = torch.argsort(base_overlap, descending=True)[:k]

            max_inlier = -1
            selected_index = -1
            selected_trans = None

            for candidate in candidates:

                trans, inliner_info, match_info = pairwise_registration_cuda(
                    base_frame.keypoints, 
                    self.frames[candidate].keypoints, 
                    base_frame.keypoints_ts,
                    self.frames[candidate].keypoints_ts,
                    base_frame.descriptors_ts, 
                    self.frames[candidate].descriptors_ts,
                    self.ird
                )

                if inliner_info > max_inlier:
                    max_inlier = inliner_info
                    selected_index = candidate
                    selected_trans = trans

            self.local_refine(selected_index, selected_trans)
            self.merge(base_frame, selected_index)

            base_overlap = torch.max(base_overlap, overlap[selected_index])
            base_mask[selected_index] = 0
            base_overlap = base_overlap * base_mask

    def collect_pose(self):
        pose = []
        for i in range(self.total_frame):
            pose.append(self.frames[i].pose)
        pose = torch.stack(pose, dim=0)
        return pose
    
    def run(self, scene_path: str):
        self.load_data(scene_path)
        
        overlap = self.get_overlap()
        self.registration(overlap)
        pose = self.collect_pose()

        return {"abs_pose_pred" : pose.cpu().numpy()}