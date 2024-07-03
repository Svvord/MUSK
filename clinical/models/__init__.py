import pytorch_lightning as pl
import torchmetrics.functional as tf
import importlib
import torch.nn.functional as F
import torch
from sksurv.metrics import concordance_index_censored
from .model_utils import get_rank

# models import
from .clam import CLAM_MB, CLAM_SB, CLAM_Batch
from .mmclassifier import MMClassifier



def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class MILModel(pl.LightningModule):
    def __init__(self, config, save_path):
        super().__init__()
        self.config = config
        self.save_path = save_path
        
        self.criterion = get_obj_from_str(
            self.config["Loss"]["name"]
            )(**self.config["Loss"]["params"])

        self.model = get_obj_from_str(config["Model"]["name"])(**config["Model"]["params"])

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_performance = 0  # save the final test performance of one fold

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch):
        patient_id, img, report, pfs, status = batch
        
        batch = (img, report, pfs, status)
        logits, result_dict = self(batch)

        loss = self.criterion(logits, pfs, status)
            
        # additional loss for vision branch
        if 'logits_vision' in result_dict.keys():
            loss += 0.3 * self.criterion(result_dict['logits_vision'], pfs, status)
        
        # additional loss for language branch
        if 'logits_report' in result_dict.keys():
            loss += 0.3 * self.criterion(result_dict['logits_report'], pfs, status)

        # Process logits by patient
        unique_patient_ids = torch.unique(patient_id)
        merged_logits = []
        merged_pfs = []
        merged_status = []

        for pid in unique_patient_ids:
            indices = (patient_id == pid).nonzero(as_tuple=True)[0]
            patient_logits = logits[indices]
            patient_status = status[indices][0]  # Assuming status is the same for all samples of the same patient
            patient_pfs = pfs[indices][0]  # Assuming pfs is the same for all samples of the same patient
            merged_logits.append(patient_logits.max())
            merged_pfs.append(patient_pfs)
            merged_status.append(patient_status)
        
        merged_logits = torch.stack(merged_logits)
        merged_pfs = torch.stack(merged_pfs)
        merged_status = torch.stack(merged_status)

        metrics = {
            "loss": loss,
            "risks": merged_logits,
            "pfs": merged_pfs,
            "status": merged_status
        }
        
        # metrics = {"loss": loss, "risks": logits, "pfs": pfs, "status": status}
        return metrics

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch)["loss"]

    def on_train_epoch_end(self):
        self.lr_scheduler.step()


    def eval_epoch(self, mode='eval'):

        step_outputs = self.validation_step_outputs if mode == 'eval' else self.test_step_outputs
        
        # gather all validation results
        risks_list = torch.cat([out["risks"] for out in step_outputs], dim=0)
        pfs_list = torch.cat([out["pfs"] for out in step_outputs], dim=0)
        status_list = torch.cat([out["status"] for out in step_outputs], dim=0)

        eval_loss = self.criterion(risks_list, pfs_list, status_list)
        self.log("val_loss", eval_loss)

        c_index = compute_c_index(risks_list, pfs_list, status_list)
        self.log("val_cindex", c_index)
        
        if get_rank() == 0:
            if mode == 'test':
                self.test_performance = c_index
                torch.save(risks_list, f'{self.save_path}/risk.pt')
                torch.save(pfs_list, f'{self.save_path}/pfs.pt')
                torch.save(status_list, f'{self.save_path}/status.pt')

            print("prob_list shape", risks_list.shape[0])
            print(f"performance c_index: {c_index:.4f}, loss: {eval_loss: .4f}")



    def validation_step(self, batch, batch_idx):
        # Compute loss and metrics for the current validation batch
        with torch.inference_mode():
            outputs = self.compute_loss(batch)
        self.validation_step_outputs.append(outputs)


    def on_validation_epoch_end(self):
        self.eval_epoch(mode='eval')
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        with torch.inference_mode():
            ret = self.compute_loss(batch)
        self.test_step_outputs.append(ret)
        return ret

    def on_test_epoch_end(self):
        self.eval_epoch(mode='test')
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        conf_optim = self.config["Optimizer"]
        name = conf_optim["optimizer"]["name"]
        optimizer_cls = getattr(torch.optim, name)
        scheduler_cls = getattr(torch.optim.lr_scheduler, conf_optim["lr_scheduler"]["name"])
        # train only trainable parameters
        optim = optimizer_cls(filter(lambda p: p.requires_grad, self.parameters()), **conf_optim["optimizer"]["params"])

        self.lr_scheduler = scheduler_cls(optim, **conf_optim["lr_scheduler"]["params"])
        return optim

def compute_c_index(risks, durations, events):

    cindex = concordance_index_censored(
        events.cpu().bool(), 
        durations.cpu(), 
        risks.squeeze().cpu(), 
        tied_tol=1e-08
        )[0]
    return cindex