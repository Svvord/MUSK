import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from copy import deepcopy
import glob
import pandas as pd
from torch.utils.data import Sampler
import numpy as np


class CoxRegDataset(Dataset):
    def __init__(self, df, config):
        """
        Initialize the CoxRegDataset.

        Parameters:
        - df (DataFrame): DataFrame containing the dataset.
        - config (dict): Configuration dictionary with 'feat_dir', 'report_dir', and 'wsi_batch' keys.
        """
        super(CoxRegDataset, self).__init__()

        self.df = df
        self.config = config
        self.feat_dir = config.get('feat_dir')
        self.report_dir = config.get('report_dir')
        self.wsi_batch = config.get('wsi_batch', False)
        self.MAX_BAG = 13000  # Maximum bag size for WSI batch

        self.feat_dict = self._load_feature_paths(self.feat_dir) if self.feat_dir else {}
        self.report_dict = self._load_feature_paths(self.report_dir) if self.report_dir else {}

    def _load_feature_paths(self, directory):
        """
        Load feature paths from the given directory.

        Parameters:
        - directory (str): Path to the directory containing feature files.

        Returns:
        - dict: Dictionary mapping image IDs to feature file paths.
        """
        file_paths = glob.glob(f"{directory}/*/*.pt")
        return {fname.split("/")[-1][:-len(".pt")]: fname for fname in file_paths}

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - tuple: Tuple containing image features, report features, time, and status.
        """
        item = self.df.iloc[idx]
        patient_id = hash(item['patient_id'])
        image_id = item.filename

        feat_image = self._load_image_features(image_id) if self.feat_dir else torch.zeros(1024)
        feat_report = self._load_report_features(image_id) if self.report_dir else torch.zeros(1024)

        time = item.time
        status = item.status

        return patient_id, feat_image, feat_report, time, status

    def _load_image_features(self, image_id):
        """
        Load image features from the feature directory.

        Parameters:
        - image_id (str): Image ID to load features for.

        Returns:
        - tensor: Tensor containing the image features.
        """
        fname = self.feat_dict.get(image_id)
        if fname:
            feat_image = torch.load(fname).float()
            if self.wsi_batch:
                if feat_image.shape[0] > self.MAX_BAG:
                    feat_image = feat_image[:self.MAX_BAG, :]
                else:
                    zero_embeddings = torch.zeros(self.MAX_BAG - feat_image.shape[0], feat_image.shape[1])
                    feat_image = torch.cat((feat_image, zero_embeddings), dim=0)
            return feat_image
        else:
            return torch.zeros(1024)

    def _load_report_features(self, image_id):
        """
        Load report features from the report directory.

        Parameters:
        - image_id (str): Image ID to load features for.

        Returns:
        - tensor: Tensor containing the report features.
        """
        fname = self.report_dict.get(image_id)
        if fname:
            feat = torch.load(fname).squeeze(0)
            return feat.float()
        else:
            return torch.zeros(1024)
        


def get_dataset_fn(dataset_name='basic'):
    assert dataset_name in ['coxreg']
    if dataset_name == 'coxreg':  # cox-regression dataset
        return CoxRegDataset
    else:
        raise NotImplementedError


class WSIDataModule(LightningDataModule):
    def __init__(self, config, split_k=0, dist=True):
        super(WSIDataModule, self).__init__()
        """
        prepare datasets and samplers
        """
        df = pd.read_csv(config["Data"]["dataframe"])

        train_index = df[df["fold"] != split_k].index
        train_df = df.loc[train_index].reset_index(drop=True)

        val_index = df[df["fold"] == split_k].index
        val_df = df.loc[val_index].reset_index(drop=True)

        # independent test cohort
        if config["Data"]["test_df"] is not None:
            test_df = pd.read_csv(config["Data"]["test_df"])
        # cross-validation test cohort; same as validation.
        else:
            test_df = deepcopy(val_df)

        dfs = [train_df, val_df, test_df]  # get training, test and validation datasets

        self.dist = dist

        # get train, val, test dataset
        dataset_name = 'basic'
        if 'dataset_name' in config['Data'].keys():
            dataset_name = config['Data']['dataset_name']
         
        self.datasets = [get_dataset_fn(dataset_name)(df, config["Data"]) for df in dfs]

        self.batch_size = config["Data"]["batch_size"]
        self.num_workers = config["Data"]["num_workers"]

    def setup(self, stage):
        
        # for training balanced sampler
        labels_list = np.array([batch[-1] for batch in self.datasets[0]])
        
        if self.dist:
            self.samplers = [DistributedSampler(dataset, shuffle=True) for dataset in self.datasets]
        else:
            # self.samplers = [BalancedSampler(labels_list), None, None]  # balanced samplers
            self.samplers = [None, None, None]  # balanced samplers

    def train_dataloader(self):
        loader = DataLoader(
            self.datasets[0],
            batch_size=self.batch_size,
            sampler=self.samplers[0],
            num_workers=self.num_workers,
            shuffle=True            
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.datasets[1],
            batch_size=self.batch_size,
            sampler=self.samplers[1],
            num_workers=self.num_workers,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.datasets[2],
            batch_size=self.batch_size,
            sampler=self.samplers[2],
            num_workers=self.num_workers,
        )
        return loader

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