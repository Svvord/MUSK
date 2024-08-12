import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from copy import deepcopy
import glob
import numpy as np
from torch.utils.data import Sampler

class BaseDataset(Dataset):
    def __init__(self, df, config):
        super(BaseDataset, self).__init__()
        self.df = df
        self.config = config
        self.feat_dir = config.get('feat_dir')
        self.report_dir = config.get('report_dir')
        self.image_key = config.get('image_key')
        self.label_key = config.get('label_key')
        
        self.feat_dict = self._create_dict(self.feat_dir) if self.feat_dir else {}
        self.report_dict = self._create_dict(self.report_dir) if self.report_dir else {}
        
        assert self.feat_dir or self.report_dir, "Both directories cannot be None!"
        
    def _create_dict(self, directory):
        return None

    def _load_features(self, item, directory, dict_key):
        if directory:
            fname = dict_key[item]
            td = torch.load(fname)
            if isinstance(td, list):
                td = torch.cat(td)
            return td.float()
        return torch.zeros(1024)
        
    def __len__(self):
        return len(self.df)
    
class CoxRegDataset(BaseDataset):
    
    def _create_dict(self, directory):
        return {os.path.basename(fname)[:-3]: fname for fname in glob.glob(f"{directory}/*/*.pt")}

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        image_id = item[self.image_key]
        patient_id = hash(image_id[:len("TCGA-A8-A0AB")])

        feat_image = self._load_features(image_id, self.feat_dir, self.feat_dict)
        feat_report = self._load_features(image_id, self.report_dir, self.report_dict)
        
        if self.feat_dir and self.config['wsi_batch']:
            MAX_BAG = 13000 # adjust this hyper-parameter based on the data distribution.
            if feat_image.shape[0] < MAX_BAG:
                zero_embeddings = torch.zeros(MAX_BAG - feat_image.shape[0], feat_image.shape[1])
                feat_image = torch.cat((feat_image, zero_embeddings), dim=0) 
            else:
                indices = torch.randperm(feat_image.shape[0])[:MAX_BAG]
                feat_image = feat_image[indices, :]

        pfs = item.time
        status = item[self.label_key]
        
        return patient_id, feat_image, feat_report, pfs, status
    
class MMDataset(BaseDataset):
    
    def _create_dict(self, directory):
        return {os.path.basename(fname)[:-3]: fname for fname in glob.glob(f"{directory}/*.pt")}
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        image_id = item[self.image_key] 
        patient_id = hash(image_id)

        feat_image = self._load_features(image_id, self.feat_dir, self.feat_dict)
        feat_report = self._load_features(image_id, self.report_dir, self.report_dict)
        
        if self.feat_dir and self.config['wsi_batch']:
            MAX_BAG = 13000
            if feat_image.shape[0] < MAX_BAG:
                zero_embeddings = torch.zeros(MAX_BAG - feat_image.shape[0], feat_image.shape[1])
                feat_image = torch.cat((feat_image, zero_embeddings), dim=0) 
            else:
                indices = torch.randperm(feat_image.shape[0])[:MAX_BAG]
                feat_image = feat_image[indices, :]

        label = torch.tensor(item[self.label_key]).long()
        
        return patient_id, feat_image, feat_report, 0, label


def get_dataset_fn(dataset_name='coxreg'):
    assert dataset_name in ['coxreg', 'multimodal']

    if dataset_name == 'coxreg': 
        return CoxRegDataset
    elif dataset_name == 'multimodal':
        return MMDataset
    else:
        raise NotImplementedError

class WSIDataModule(LightningDataModule):
    def __init__(self, config, split_k=0, dist=True):
        super(WSIDataModule, self).__init__()
        
        if config['Data']['test_df'] is None:
            df = pd.read_csv(config["Data"]["train_df"])
            train_df = df[df["fold"] != split_k].reset_index(drop=True)
            val_df = df[df["fold"] == split_k].reset_index(drop=True)
            test_df = pd.read_csv(config["Data"]["test_df"]) if config["Data"]["test_df"] else deepcopy(val_df)
        else:
            train_df = pd.read_csv(config["train_df"])
            val_df = pd.read_csv(config["val_df"])
            test_df = pd.read_csv(config["test_df"])
        
        self.dist = dist
        dataset_name = config["Data"].get('dataset_name', 'basic')
        self.datasets = [get_dataset_fn(dataset_name)(df, config["Data"]) for df in [train_df, val_df, test_df]]
        
        self.batch_size = config["Data"]["batch_size"]
        self.num_workers = config["Data"]["num_workers"]
        
    def setup(self, stage=None):
        if self.dist:
            self.samplers = [DistributedSampler(dataset, shuffle=True) for dataset in self.datasets]
        else:
            labels_list = np.array([item[-1] for item in self.datasets[0]])
            self.samplers = [BalancedSampler(labels_list), None, None]
            
    def train_dataloader(self):
        return DataLoader(
            self.datasets[0],
            batch_size=self.batch_size,
            sampler=self.samplers[0],
            num_workers=self.num_workers,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[1],
            batch_size=self.batch_size,
            sampler=self.samplers[1],
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets[2],
            batch_size=self.batch_size,
            sampler=self.samplers[2],
            num_workers=self.num_workers
        )


class BalancedSampler(Sampler):
    def __init__(self, dataset_labels):
        self.indices = []
        self.num_samples = 0

        # Create a list of indices for each class
        label_to_indices = {label: np.where(dataset_labels == label)[0] for label in np.unique(dataset_labels)}

        # Find the maximum size among the classes to balance
        largest_class_size = max(len(indices) for indices in label_to_indices.values())

        # Extend indices of smaller classes by sampling with replacement
        for indices in label_to_indices.values():
            indices_balanced = np.random.choice(indices, largest_class_size, replace=True)
            self.indices.append(indices_balanced)
            self.num_samples += largest_class_size

        # Flatten list and shuffle
        self.indices = np.random.permutation(np.hstack(self.indices))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples
    