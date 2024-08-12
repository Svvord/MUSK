import pytorch_lightning as pl
import torchmetrics.functional as tf
import importlib
import torch.nn.functional as F
import torch
from sksurv.metrics import concordance_index_censored

from .model_utils import get_rank

# models import
from .abmil import AbMIL
from .clam import CLAM_MB, CLAM_SB, CLAM_Batch
from .mmclassifier import MMClassifier


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)



def pred_patient(patient_id, logits, status, pfs):
    # Assuming patient_id is a PyTorch tensor
    unique_patient_ids = []
    seen = set()

    for pid in patient_id:
        if pid.item() not in seen:
            unique_patient_ids.append(pid.item())
            seen.add(pid.item())

    # Convert back to a tensor if needed
    unique_patient_ids = torch.tensor(unique_patient_ids)

    # Lists to store merged logits, pfs, and status
    merged_logits = []
    merged_pfs = []
    merged_status = []

    # Loop through each unique patient ID
    for pid in unique_patient_ids:
        # Get indices for the current patient
        indices = (patient_id == pid).nonzero(as_tuple=True)[0]
        
        # Get logits, status, and pfs for the current patient
        patient_logits = logits[indices]
        patient_status = status[indices][0]  
        patient_pfs = pfs[indices][0]  
        
        merged_logit = patient_logits.mean()
        
        # Append merged values to respective lists
        merged_logits.append(merged_logit)
        merged_pfs.append(patient_pfs)
        merged_status.append(patient_status)

    # Stack merged values into tensors
    merged_logits = torch.stack(merged_logits)
    merged_pfs = torch.stack(merged_pfs)
    merged_status = torch.stack(merged_status)

    return merged_logits, merged_pfs, merged_status




class MILModel(pl.LightningModule):
    def __init__(self, config, save_path):
        super().__init__()
        self.config = config
        self.save_path = save_path
        self.lr_scheduler = None
        
        self.criterion = get_obj_from_str(
            self.config["Loss"]["name"]
            )(**self.config["Loss"]["params"])

        # determin whether we are doing classification or regression        
        cox_loss = ['NLLLogistiHazardLoss', 'CoxPHLoss', 'CoxSurvLoss']
        if self.criterion.__class__.__name__ in cox_loss:
            self.task = 'reg'
        else:
            self.task = 'cls'
        
        self.model = get_obj_from_str(config["Model"]["name"])(**config["Model"]["params"])
        self.test_performance = 0

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch):

        if self.task == "cls":
            patient_id, img, report, _, y = batch
            logits, result_dict = self((img, report))
            inst_loss = result_dict.get("instance_loss", torch.tensor(0.0, device=self.device))
                        
            if self.criterion.__class__.__name__ == "FocalLoss":
                loss = self.criterion(torch.nn.Sigmoid()(logits[:, 1]), y) + 0.1 * inst_loss
            else:
                loss = self.criterion(logits, y) + 0.1 * inst_loss
            
            
            if 'logits_vision' in result_dict.keys():
                if self.criterion.__class__.__name__ == "FocalLoss":
                    loss += self.criterion(torch.nn.Sigmoid()(result_dict['logits_vision'][:,1]), y)
                else:
                    loss += self.criterion(result_dict['logits_vision'], y)

            if 'logits_report' in result_dict.keys():
                if self.criterion.__class__.__name__ == "FocalLoss":
                    loss += 0.1 * self.criterion(torch.nn.Sigmoid()(result_dict['logits_report'][:,1]), y)
                else:
                    loss += 0.1 * self.criterion(result_dict['logits_report'], y)
                                                 

            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = torch.nn.functional.softmax(logits, dim=1)
            metrics = {"loss": loss, "preds": Y_hat, "probs": Y_prob, "labels": y, "logits": logits}

        elif self.task == "reg":            
            patient_id, img, report, pfs, status = batch
            logits, result_dict = self((img, report))
            loss = self.criterion(logits, pfs, status)
            
            if 'logits_vision' in result_dict.keys():
                loss += 0.1 * self.criterion(result_dict['logits_vision'], pfs, status)

            if 'logits_report' in result_dict.keys():
                loss += 0.1 * self.criterion(result_dict['logits_report'], pfs, status)
                
            metrics = {"patient_id": patient_id, "loss": loss, "risks": logits, "pfs": pfs, "status": status}
        else:
            raise NotImplementedError
        return metrics


    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch)["loss"]

    def on_train_epoch_end(self):
        self.lr_scheduler.step()


    def eval_epoch(self, mode='eval'):

        step_outputs = self.validation_step_outputs if mode == 'eval' else self.test_step_outputs

        if self.task == 'cls':
                
            # gather all validation results
            pred_list = torch.cat([out["preds"] for out in step_outputs], dim=0)
            label_list = torch.cat([out["labels"] for out in step_outputs], dim=0)
            prob_list = torch.cat([out["probs"] for out in step_outputs], dim=0)
            logit_list = torch.cat([out["logits"] for out in step_outputs], dim=0)
            

            if self.criterion.__class__.__name__ == "FocalLoss":
                eval_loss = self.criterion(torch.nn.Sigmoid()(logit_list[:, 1]), label_list)
            else:
                eval_loss = self.criterion(logit_list, label_list)
                
            self.log("val_loss", eval_loss)

            acc = tf.accuracy(pred_list, label_list)
            auc = tf.auroc(prob_list[:, 1], label_list)
            self.log("val_auc", auc)
            
            if mode == "test":
                self.test_performance = auc.cpu().item()
                torch.save(pred_list, f'{self.save_path}/pred.pt')
                torch.save(label_list, f'{self.save_path}/label.pt')
                torch.save(prob_list[:, 1], f'{self.save_path}/prob.pt')
                torch.save(logit_list, f'{self.save_path}/logit.pt')

            print("prob_list shape", prob_list.shape[0])
            print(f"performance acc: {acc:.4f}, auc: {auc:.4f} loss: {eval_loss: .4f}")

        elif self.task == 'reg':
            
            # gather all validation results
            risks_list = torch.cat([out["risks"] for out in step_outputs], dim=0)
            pfs_list = torch.cat([out["pfs"] for out in step_outputs], dim=0)
            status_list = torch.cat([out["status"] for out in step_outputs], dim=0)
            patient_list = torch.cat([out["patient_id"] for out in step_outputs], dim=0)

            # merge slide-level predictions
            risks_list, pfs_list, status_list = pred_patient(patient_list, risks_list, status_list, pfs_list)

            c_index = compute_c_index(risks_list, pfs_list, status_list)
            self.log("val_cindex", c_index)

            eval_loss = self.criterion(risks_list, pfs_list, status_list)
            self.log("val_loss", eval_loss)
    
            if mode == 'test':
                self.test_performance = c_index
                torch.save(risks_list, f'{self.save_path}/risk.pt')
                torch.save(pfs_list, f'{self.save_path}/pfs.pt')
                torch.save(status_list, f'{self.save_path}/status.pt')

            print("prob_list shape", risks_list.shape[0])
            print(f"performance c_index: {c_index:.4f}, loss: {eval_loss: .4f}")

        else:
            raise NotImplementedError


    def validation_step(self, batch, batch_idx):
        
        # Compute loss and metrics for the current validation batch
        with torch.inference_mode():
            outputs = self.compute_loss(batch)
        self.validation_step_outputs.append(outputs)
        return outputs

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
        optim = optimizer_cls(self.parameters(), **conf_optim["optimizer"]["params"])
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