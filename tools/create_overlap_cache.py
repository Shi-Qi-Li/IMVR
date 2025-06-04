import os
import sys
import pickle
from tqdm import tqdm
sys.path.append(os.getcwd())

from dataset import build_dataset, build_dataloader
from utils import set_random_seed

def cache_overlap_data(total_epoch: int, desc: str):
    set_random_seed(3407)

    train_set = build_dataset({
        "name": "Scene",
        "data_path": "data",
        "split": "train",
        "descriptor": desc,
        "ird": 0.07,
        "point_sample": True,
        "frame_sample": True,
        "frame_limit": [8, 60],
        "point_limit": 5000,
        "overlap_only": True,
        "processes": 1
    })

    train_loader = build_dataloader(
        train_set, 
        True, 
        {
            "num_workers": 0,
            "batch_size": 1
        }
    )

    save_path = f"data/vlad_{desc}_cache"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for epoch in range(total_epoch):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        loop.set_description(f'Epoch [{epoch}/{total_epoch}]')
        for idx, data_batch in loop:

            data_batch["descriptors"] = [descriptor.squeeze(0).detach().cpu().numpy() for descriptor in data_batch["descriptors"]]
            data_batch["gt_overlap"] = data_batch["gt_overlap"].squeeze(0).detach().cpu().numpy()

            with open(os.path.join(save_path, f"{epoch}_{idx}.pkl"), "wb") as f:
                pickle.dump(data_batch, f)

if __name__ == "__main__":
    cache_overlap_data(300, "yoho")