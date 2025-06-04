import os
import pickle
from torch.utils.data import Dataset
import numpy as np
import numpy.typing as npt


from .builder import DATASET

@DATASET
class Cache(Dataset):
    def __init__(self, data_path: str, cache_folder: str, split: str, total_epoch: int):
        super().__init__()
        self.data_path = data_path
        self.cache_folder = cache_folder
        
        self.split = split
        self.scene_num = 48 if split == "train" else 6

        self.data_list = [f"{epoch}_{index}" for index in range(self.scene_num) for epoch in range(total_epoch)]
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        pickle_path = os.path.join(self.data_path, self.cache_folder, f"{self.data_list[index]}.pkl")

        with open(pickle_path, 'rb') as f:
            data_batch = pickle.load(f)

        data_batch["registration_overlap_ratio"] = data_batch["registration_overlap_ratio"] + np.eye(data_batch["registration_overlap_ratio"].shape[0], dtype=np.float32)

        return data_batch