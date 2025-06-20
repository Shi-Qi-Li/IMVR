from torch.utils.data import DataLoader

from utils import Registry

DATASET = Registry("dataset")

def build_dataset(cfg):
    return DATASET.build(cfg)

def build_dataloader(dataset, shuffle, cfg, sampler = None):
    cfg["dataset"] = dataset
    cfg["shuffle"] = shuffle
    if "collate_fn" in cfg:
        cfg["collate_fn"] = eval(cfg["collate_fn"])
    if sampler is not None:
        cfg["sampler"] = sampler
    return DataLoader(**cfg)
