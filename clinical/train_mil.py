import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from wsi_dataset import WSIDataModule
import yaml
from models import MILModel
import os
import random
import numpy as np
import json
import glob 

def read_yaml(fpath):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return dict(yml)

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

if __name__ == "__main__":
    # Configs
    fname = "config_prognosis_musk"
    config_yaml = read_yaml(f"./configs/{fname}.yaml")
    for key, value in config_yaml.items():
        print(f"{key.ljust(30)}: {value}")

    num_gpus = 1
    dist = False
    N_SPLITS = 5
    seed = 42

    ret = dict()
    fix_seed(seed)
    
    # get all training projects
    df_path = config_yaml['Data']['train_df'] 
    if os.path.isdir(df_path):
        dfs = list(glob.glob(f"{df_path}/*.csv"))
    elif os.path.isfile(df_path):
        dfs = [df_path]
    
    for df in dfs:

        project = os.path.basename(df)[:-len(".csv")]
        config_yaml['Data']['train_df'] = df
        rets_fold = []

        for split_k in range(N_SPLITS):

            workspace = f"outputs_{fname}"
            os.makedirs(workspace, exist_ok=True)
            dm = WSIDataModule(config_yaml, split_k=split_k, dist=dist)

            resume_path = None
            save_path = f"./workspace/models/{fname}/{project}/seed{seed}/fold_{split_k}"
            os.makedirs(save_path, exist_ok=True)

            # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
            model = MILModel(config_yaml, save_path=save_path)
            save_fname = "{epoch}-{val_auc:.3f}-{val_loss:.3f}" if model.task == 'cls' else "{epoch}-{val_cindex:.3f}-{val_loss:.3f}"
            monitor = "val_auc" if model.task == 'cls' else "val_cindex"

            checkpoint_cb = ModelCheckpoint(save_top_k=1,
                                            monitor=monitor,
                                            mode="max",
                                            dirpath=workspace,
                                            verbose=True,
                                            filename=save_fname)

            early_stop_callback = EarlyStopping(
                monitor=monitor, 
                min_delta=0.00, 
                patience=10, 
                mode="max"
                )

            trainer = pl.Trainer(
                accelerator="gpu",
                devices=num_gpus,
                deterministic=False,
                precision=16,
                callbacks=[checkpoint_cb, early_stop_callback],
                max_epochs=config_yaml["General"]["epochs"],
                accumulate_grad_batches=config_yaml["General"]["acc_steps"], 
                logger=False
            )

            # Train!
            trainer.fit(model, datamodule=dm, ckpt_path=resume_path)
            
            # test
            wts = trainer.checkpoint_callback.best_model_path  # get the best checkpoint
            trainer.test(model, datamodule=dm, ckpt_path=wts)
            rets_fold.append(model.test_performance)
            
            # save the best model
            torch.save(torch.load(wts)['state_dict'], f'{save_path}/fold_{split_k}.pth')
            os.system(f"rm -rf {workspace}")
        
        # >>>>>>>>>>>>> final results of cross-validation >>>>>>>>>>>>> #
        macro_avg = np.mean(rets_fold)
        macro_std = np.std(rets_fold)
        ret_fold_list = [round(float(value), 4) for value in rets_fold] 
        print(f"Macro-avg cindex: {macro_avg:.4f} Macro-std cindex: {macro_std:.4f}")
        
        # store results
        ret = {project:{
            "seed": seed,
            "macro_avg": round(macro_avg, 4),
            "macro_std": round(macro_std, 4),
            "ret_folds": ret_fold_list
            }
        }
        
        with open(f"./workspace/results/{fname}.json", "a+") as f:
            json.dump(ret, f)
            f.write('\n')






