import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from wsi_dataset import WSIDataModule
import yaml
import importlib
from models import MILModel
import os
import random
import numpy as np
import torchmetrics.functional as tf
from models import compute_c_index
import glob
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

def read_yaml(fpath):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return dict(yml)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


# seed everything
def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)  # torch < 1.8
    torch.use_deterministic_algorithms(True, warn_only=True)  # torch >= 1.8


def create_fold(N_SPLITS, fname, temp_dir, seed):
    # Read data
    df = pd.read_csv(fname)

    # Create a composite key 'group_key' for grouping and stratification; ensure the same patient is in the same group.
    df['group_key'] = df['patient_id'].astype(str) + '_' + df['status'].astype(str)

    # Shuffle the DataFrame if necessary
    np.random.seed(seed)  # Set seed for reproducibility
    df = df.sample(frac=1).reset_index(drop=True)

    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=N_SPLITS)

    # Split data ensuring the same patient and similar status stays in the same fold
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df['group_key'])):
        df.loc[test_idx, 'fold'] = fold

    # Reset index and save the dataframe with fold information
    df.to_csv(f"{temp_dir}/temp.csv")


def read_config(fname):
    return read_yaml(f"./configs/{fname}.yaml")

def setup_workspace(workspace):
    os.makedirs(workspace, exist_ok=True)

def get_project_list():
    return sorted(glob.glob("./workspace/splits/*.csv"))

def prepare_trainer(config_yaml, num_gpus, workspace, monitor="val_cindex"):
    save_fname = "{epoch}-{val_cindex:.3f}-{val_loss:.3f}"
    checkpoint_cb = ModelCheckpoint(
        save_top_k=1,
        monitor=monitor,
        mode="max",
        dirpath=workspace,
        verbose=True,
        filename=save_fname
    )
    
    early_stop_cb = EarlyStopping(
        monitor=monitor, 
        min_delta=0.00, 
        patience=10, 
        mode="max"
    )
    
    return pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        deterministic=False,
        precision=16,
        callbacks=[checkpoint_cb, early_stop_cb],
        max_epochs=config_yaml["General"]["epochs"],
        accumulate_grad_batches=config_yaml["General"]["acc_steps"], 
        logger=False
    )

def save_results(results, fname):
    results_dir = "./workspace/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{fname}.json")
    with open(results_file, "a+") as f:
        json.dump(results, f)
        f.write('\n')

def main():
    fname = "config_clam_coxreg_musk"
    config_yaml = read_config(fname)
    for key, value in config_yaml.items():
        print(f"{key.ljust(30)}: {value}")

    num_gpus = 1
    dist = False
    N_SPLITS = 5
    seed_list = [42]
    all_projs = get_project_list()

    for proj in all_projs:
        config_yaml['Data']['dataframe'] = proj
        original_csv = config_yaml['Data']['dataframe']
        proj_name = os.path.basename(proj).replace(".csv", "")

        for seed in seed_list:
            fix_seed(seed)
            workspace = f"outputs_{fname}"
            rets_fold = []  # save the performance of each fold

            for split_k in range(N_SPLITS):
                setup_workspace(workspace)
                create_fold(N_SPLITS, original_csv, workspace, seed)
                config_yaml['Data']['dataframe'] = f"{workspace}/temp.csv"

                dm = WSIDataModule(config_yaml, split_k=split_k, dist=dist)
                save_path = f"./workspace/models/{fname}/{proj_name}/seed{seed}/fold_{split_k}"
                setup_workspace(save_path)

                model = MILModel(config_yaml, save_path=save_path, encoder=None)
                trainer = prepare_trainer(config_yaml, num_gpus, workspace)

                trainer.fit(model, datamodule=dm, ckpt_path=None)
                wts = trainer.checkpoint_callback.best_model_path
                trainer.test(model, datamodule=dm, ckpt_path=wts)
                rets_fold.append(model.test_performance)

                torch.save(torch.load(wts)['state_dict'], f'{save_path}/fold_{split_k}.pth')
                os.system(f"rm -rf {workspace}")

            macro_avg = np.mean(rets_fold)
            macro_std = np.std(rets_fold)
            results = {
                proj_name: {
                    "seed": seed,
                    "macro_cindex": round(macro_avg, 4),
                    "macro_std": round(macro_std, 4),
                    "folds": rets_fold
                }
            }
            save_results(results, fname)

if __name__ == "__main__":
    main()
