import torch
import torch.nn as nn
import importlib
from einops import rearrange


"""
We propose to use vision-language features for Outcome prediction.
"""


def get_obj_from_str(image_mil_name, reload=False):
    module, cls = image_mil_name.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        # x of shape [batch_size, feat_dim]
        cls_rep = self.norm(x)
        pooled_output = self.dense(cls_rep)
        # pooled_output = self.activation(pooled_output)
        return pooled_output


class MMClassifier(nn.Module):
    def __init__(self, image_mil_name, mil_params, feat_dim, num_classes):
        super(MMClassifier, self).__init__()
        
        target_dim = mil_params['hidden_feat']

        self.image_mil = get_obj_from_str(image_mil_name)(
            feat_dim=feat_dim,
            n_classes=num_classes,
            **mil_params
            )
        self.feat_dim = feat_dim

        self.fc_vision = Pooler(target_dim, target_dim)
        self.fc_report = Pooler(feat_dim, target_dim) 
        
        self.classifier_vision = nn.Linear(target_dim, num_classes)
        self.classifier_report = nn.Linear(target_dim, num_classes)
        self.classifier_final = nn.Linear(target_dim, num_classes)

    def forward(self, batch):
        images, reports = batch
        
        # aggregate wsi; we do not use clustering loss, thus it is an ABMIL moddel. 
        feat_vision, _ = self.image_mil(
            (images, None), 
            return_global_feature=True
            )
        
        feat_vision = self.fc_vision(feat_vision)
       
        # aggregate report
        feat_report = self.fc_report(reports)
        
        results_dict = dict()
        global_feat = feat_vision + feat_report

        # additional loss
        logits_vision = self.classifier_vision(feat_vision)                
        logits_report = self.classifier_report(feat_report)
        results_dict.update({"logits_vision": logits_vision, "logits_report": logits_report})

        logits = self.classifier_final(global_feat)
        
        return logits, results_dict
