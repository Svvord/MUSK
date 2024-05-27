import os
import glob
import random

import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import pandas as pd
import h5py
import numpy
import json
import pandas as pd


class PCamDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        if train:
            self.data_x = h5py.File(os.path.join(root, "camelyonpatch_level_2_split_train_x.h5"), "r")['x']
            self.data_y = h5py.File(os.path.join(root, "camelyonpatch_level_2_split_train_y.h5"), "r")['y']
        else:
            self.data_x = h5py.File(os.path.join(root, "camelyonpatch_level_2_split_test_x.h5"), "r")['x']
            self.data_y = h5py.File(os.path.join(root, "camelyonpatch_level_2_split_test_y.h5"), "r")['y']

        self.trans = transform

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.classes = ['Lymph node', 'Lymph node containing metastatic tumor tissue']

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, item):
        image = Image.fromarray(self.data_x[item])
        label = self.data_y[item]

        if self.trans is not None:
            if self.trans.__class__.__name__ == "CLIPProcessor":
                image = self.trans(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.trans(image)

        label = int(numpy.squeeze(label))
        return image, label


class NCTCRCDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        self.trans = transform
        self.cancer_list = ["STR", "TUM"]
        self.normal_list = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM"]
        if train:
            self.data = list(glob.glob(f"{root}/NCT-CRC-HE-100K/*/*.tif"))
        else:
            self.data = list(glob.glob(f"{root}/CRC-VAL-HE-7K/*/*.tif"))

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

        # self.classes = ['Adipose', 'Debris', 'Lymphocytes', 'Mucus', 'Smooth muscle',
        #                 'Normal colon mucosa',
        #                 'Cancer-associated stroma',
        #                 'Colorectal adenocarcinoma epithelium']
        self.classes = [
            "Adipose", "background", "debris", "lymphocytes", "mucus", "smooth muscle",
            "normal colon mucosa", "cancer - associated stroma", "colorectal adenocarcinoma epithelium"]

        class2label_binary = {"ADI": 0, "BACK": 0, "DEB": 0, "LYM": 0, "MUC": 0, "MUS": 0, "NORM": 0, "STR": 1,
                              "TUM": 1}
        class2label_multiclass = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7,
                                  "TUM": 8}

        self.class2label = class2label_multiclass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = Image.open(self.data[item]).convert("RGB")

        if self.trans is not None:
            if self.trans.__class__.__name__ == "CLIPProcessor":
                image = self.trans(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.trans(image)

        image_path = self.data[item]
        label_name = image_path.split("/")[-2]
        label = self.class2label[label_name]

        # label = 0
        #
        # for element in self.cancer_list:
        #     if element in self.data[item]:
        #         label = 1

        return image, label


class MhistDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, image_dir, transform=None, train=True):
        csv_file = os.path.join(root, csv_file)
        image_dir = os.path.join(root, image_dir)

        self.data = pd.read_csv(csv_file)
        if train:
            self.data = self.data[self.data['Partition'] == 'train']
        else:
            self.data = self.data[self.data['Partition'] != 'train']
        self.image_paths = self.data['Image Name'].values
        self.labels = self.data['Majority Vote Label'].values
        self.image_dir = image_dir
        self.transform = transform
        self.train = train
        self.cat_to_num_map = {'HP': 0, 'SSA': 1}
        self.classes = ["hyperplastic polyp", "sessile serrated adenoma"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.cat_to_num_map[self.labels[index]]

        return image, label


class SicapDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_dir, transform=None, train=True):

        image_dir = os.path.join(root, image_dir)

        if train:
            csv_file = os.path.join(root, "partition/Test", "Train.xlsx")
            self.data = pd.read_excel(csv_file)
        else:
            csv_file = os.path.join(root, "partition/Test", "Test.xlsx")
            self.data = pd.read_excel(csv_file)

        # drop all columns except image_name and the label columns
        label_columns = ['NC', 'G3', 'G4', 'G5']  # , 'G4C']
        self.data = self.data[['image_name'] + label_columns]

        # get the index of the maximum label value for each row
        self.data['labels'] = self.data[label_columns].idxmax(axis=1)

        # replace the label column values with categorical values
        self.cat_to_num_map = label_map = {'NC': 0, 'G3': 1, 'G4': 2, 'G5': 3}  # , 'G4C': 4}
        self.data['labels'] = self.data['labels'].map(label_map)

        self.image_paths = self.data['image_name'].values
        self.labels = self.data['labels'].values
        self.image_dir = image_dir
        self.transform = transform
        self.train = train

        # these prompts work better!
        self.classes = ["non-cancerous well-differentiated glands",
                        "gleason grade 3 with atrophic well differentiated and dense glandular regions",
                        "gleason grade 4 with cribriform, ill-formed, large-fused and papillary glandular patterns",
                        "gleason grade 5 with nests of cells without lumen formation, isolated cells and pseudo-roseting patterns",
                        ]

        # self.classes = ['Benign glands',
        #                 'Atrophic dense glands',
        #                 'Cribriform ill-formed fused papillary patterns',
        #                 'Isolated nest cellswithout lumen rosetting patterns']

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels[index]

        return image, label


class ArchCsvDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transforms, img_key='image_path', caption_key='caption', sep=","):
        df = pd.read_csv(csv_file, sep=sep)
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.ids = list(sorted(df['ids'].tolist()))

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        image = Image.open(str(self.images[id_])).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                images = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                images = self.transform(image)

        texts = [str(self.captions[id_])]
        return images, texts


class OsteoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        self.root = root

        if train:
            self.data = json.load(open(os.path.join(root, "train.json")))
        else:
            self.data = json.load(open(os.path.join(root, "test.json")))

        self.csv_file = pd.read_csv(os.path.join(root, "labels.csv"), index_col=False, header=None)

        self.transform = transform
        self.cat_to_num_map = {'Non-Tumor': 0, 'Non-Viable-Tumor': 1, 'Viable': 2}
        self.classes = ["non-tumor", "non-viable necrotic osteosarcoma tumor", "viable osteosarcoma tumor"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.data[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        class_name = self.csv_file[self.csv_file[2] == image_path.split("/")[-1]][1]
        label = self.cat_to_num_map[class_name.iloc[0]]

        return image, label


class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transform=None, train=True, val=False,
                 tumor=False):
        csv_file = os.path.join(root, csv_file)
        self.data = pd.read_csv(csv_file)
        self.data_root = root

        if train:
            self.data = self.data[self.data['set'] == 'Train']
        else:
            if val:
                self.data = self.data[self.data['set'] == "Validation"]
            else:
                self.data = self.data[self.data['set'] == 'Test']

        if tumor:
            self.data = self.data[self.data['malignicy'] == 'tumor']
        self.tumor = tumor

        self.image_paths = self.data['file'].values
        self.labels = self.data['class'].values

        self.transform = transform
        self.train = train

        self.cat_to_num_map = {'nontumor_skin_necrosis_necrosis': 0,
                               'nontumor_skin_muscle_skeletal': 1,
                               'nontumor_skin_sweatglands_sweatglands': 2,
                               'nontumor_skin_vessel_vessel': 3,
                               'nontumor_skin_elastosis_elastosis': 4,
                               'nontumor_skin_chondraltissue_chondraltissue': 5,
                               'nontumor_skin_hairfollicle_hairfollicle': 6,
                               'nontumor_skin_epidermis_epidermis': 7,
                               'nontumor_skin_nerves_nerves': 8,
                               'nontumor_skin_subcutis_subcutis': 9,
                               'nontumor_skin_dermis_dermis': 10,
                               'nontumor_skin_sebaceousglands_sebaceousglands': 11,
                               'tumor_skin_epithelial_sqcc': 12,
                               'tumor_skin_melanoma_melanoma': 13,
                               'tumor_skin_epithelial_bcc': 14,
                               'tumor_skin_naevus_naevus': 15
                               }

        self.tumor_map = {'tumor_skin_epithelial_sqcc': 0,
                          'tumor_skin_melanoma_melanoma': 1,
                          'tumor_skin_epithelial_bcc': 2,
                          'tumor_skin_naevus_naevus': 3
                          }

        self.classes = list(self.cat_to_num_map) if not self.tumor else list(self.tumor_map)

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(os.path.join(self.data_root, image_path)).convert('RGB')

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        if not self.tumor:
            label = self.cat_to_num_map[self.labels[index]]
        else:
            label = self.tumor_map[self.labels[index]]

        return image, label


class WSSS4LUADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        json_file = json.load(open(os.path.join(root, "data_split.json")))

        self.data = []
        split = 'train' if train else 'test'
        for item in json_file:
            if item['split'] == split:
                self.data.append(item)

        self.image_dir = root
        self.transform = transform
        self.train = train

        # these prompts work better!
        self.classes = ["benign",
                        "cancer"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.data[index]['path'])
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.data[index]['label']

        return image, label


class PannukeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        self.root = root

        df = pd.read_csv(os.path.join(root, "PanNuke_all_binary.csv"))
        self.df = df[df['split'] == 'train'] if train else df[df['split'] == 'test']

        self.transform = transform

        self.classes = ["benign",
                        "malignant"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.df.iloc[index]['image'])
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = 1 if 'malignant' in self.df.iloc[index]['caption'] else 0
        return image, label


class PannukeRetrievalDataset(torch.utils.data.Dataset):
    """
    for image retrieval
    """

    def __init__(self, root, transform=None):
        self.df = pd.read_csv(os.path.join(root, "PanNuke_all_binary.csv"))
        self.transform = transform

        self.classes = ['bile-duct', 'liver', 'breast', 'pancreatic', 'adrenal gland', 'cervix', 'skin', 'kidney',
                        'lung', 'thyroid', 'esophagus', 'uterus', 'bladder', 'prostate', 'testis',
                        'ovarian', 'stomach', 'headneck', 'colon']

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fpath = self.df.iloc[index]['image']
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label_name = fpath.split("/")[-1].split("_")[0]
        label = self.classes.index(label_name)
        return image, label


class UnitopathoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        if train:
            self.data = json.load(open(os.path.join(root, "images_train.json")))
        else:
            self.data = json.load(open(os.path.join(root, "images_test.json")))
        self.root = root
        self.transform = transform

        self.labels_dict = {"HP": 0,
                            "NORM": 1,
                            "TA.HG": 2,
                            "TA.LG": 3,
                            "TVA.HG": 4,
                            "TVA.LG": 5}
        # NORM - Normal
        # tissue;
        # HP - Hyperplastic
        # Polyp;
        # TA.HG - Tubular
        # Adenoma, High - Grade
        # dysplasia;
        # TA.LG - Tubular
        # Adenoma, Low - Grade
        # dysplasia;
        # TVA.HG - Tubulo - Villous
        # Adenoma, High - Grade
        # dysplasia;
        # TVA.LG - Tubulo - Villous
        # Adenoma, Low - Grade
        # dysplasia.

        self.classes = ["Hyperplastic Polyp",
                        "Normal tissue",
                        "Tubular Adenoma, High-Grade dysplasia",
                        "Tubular Adenoma, Low-Grade dysplasia",
                        "Tubulo-Villous Adenoma, High-Grade dysplasia",
                        "Tubulo-Villous Adenoma, Low-Grade dysplasia"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.data[index])
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[fpath.split("/")[-2]]

        return image, label


class UnitopathoRetrievalDataset(torch.utils.data.Dataset):
    """
    Dataset for unitopatho image retrieval, using all samples.
    """

    def __init__(self, root, transform=None, train=True):
        fp1 = json.load(open(os.path.join(root, "images_train.json")))
        fp2 = json.load(open(os.path.join(root, "images_test.json")))
        self.data = fp1 + fp2

        self.root = root
        self.transform = transform

        self.labels_dict = {"HP": 0,
                            "NORM": 1,
                            "TA.HG": 2,
                            "TA.LG": 3,
                            "TVA.HG": 4,
                            "TVA.LG": 5}

        # these prompts work better!
        self.classes = ["HP",
                        "NORM",
                        "TA.HG",
                        "TA.LG",
                        "TVA.HG",
                        "TVA.LG"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.data[index])
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[fpath.split("/")[-2]]

        return image, label


class LC25ColonDataset(torch.utils.data.Dataset):
    """
    Dataset for LC25000 colon image zero-shot retrieval
    """

    def __init__(self, root, transform=None, train=True):
        if train:
            self.data = json.load(open(os.path.join(root, "train.json"), "r"))
        else:
            self.data = json.load(open(os.path.join(root, "test.json"), "r"))

        self.root = root
        self.transform = transform

        self.labels_dict = {"colon_aca": 0,
                            "colon_n": 1}

        self.classes = ["colon_aca",
                        "colon_n"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.data[index])
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[fpath.split("/")[-2]]

        return image, label


class LC25LungDataset(torch.utils.data.Dataset):
    """
    Dataset for LC25000 lunge image zero-shot retrieval
    """

    def __init__(self, root, transform=None, train=True):
        if train:
            self.data = json.load(open(os.path.join(root, "train.json"), "r"))
        else:
            self.data = json.load(open(os.path.join(root, "test.json"), "r"))

        self.root = root
        self.transform = transform

        self.labels_dict = {"lung_aca": 0,
                            "lung_n": 1,
                            "lung_scc": 2}

        self.classes = ["lung_aca",
                        "lung_n",
                        "lung_scc"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.data[index])
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[fpath.split("/")[-2]]

        return image, label


class LC25Dataset(torch.utils.data.Dataset):
    """
    Dataset for LC25000 image linear probe
    """

    def __init__(self, root, transform=None, train=True):
        if train:
            self.data = json.load(open(os.path.join(root, "train.json"), "r"))
        else:
            self.data = json.load(open(os.path.join(root, "test.json"), "r"))

        self.root = root
        self.transform = transform

        self.labels_dict = {"colon_aca": 0,
                            "colon_n": 1,
                            "lung_aca": 2,
                            "lung_n": 3,
                            "lung_scc": 4}
        # colon adenocarcinoma, benign colonic tissue, lung adenocarcinoma,
        # lung squamous cell carcinoma, and benign lung tissue.

        self.classes = ["colon adenocarcinoma",
                        "benign colonic tissue",
                        "lung adenocarcinoma",
                        "benign lung tissue",
                        "lung squamous cell carcinoma"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.data[index])
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[fpath.split("/")[-2]]

        return image, label


class RenalCellDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, train=True):
        if train:
            self.data = json.load(open(os.path.join(root, "train.json"), "r"))
        else:
            self.data = json.load(open(os.path.join(root, "test.json"), "r"))

        self.root = root
        self.transform = transform

        self.labels_dict = {"blood": 0,
                            "cancer": 1,
                            "empty": 2,
                            "normal": 3,
                            "other": 4,
                            "stroma": 5}

        self.classes = ["blood",
                        "cancer",
                        "empty",
                        "normal",
                        "other",
                        "stroma"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.data[index])
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[fpath.split("/")[-2]]

        return image, label


class BRACS6ClsDataset(torch.utils.data.Dataset):
    """
    BRACS dataset ROIs are scanned in 40x; and the image size is around 1800*1800 - 2400*2400;
    we get 224*224 at 5x; thus, the crop_size = 224*8.
    """
    def __init__(self, root, transform=None, train=True, is_retrieval=False):

        self.transform = transform
        self.root = root

        self.crop_size = 224 * 8
        df = pd.read_csv(os.path.join(root, "bright_roi.csv"))

        if not is_retrieval:
            # take data split for classification task
            if train:
                self.df = df[df['data_split'] == 'train']
            else:
                self.df = df[df['data_split'] == 'test']
        else:
            # use all images for image retrieval task
            self.df = df

        self.labels_dict = {"0_N": 0,
                            "1_PB": 0,
                            "2_UDH": 1,
                            "3_FEA": 2,
                            "4_ADH": 3,
                            "5_DCIS": 4,
                            "6_IC": 5
                            }

        self.classes = ["normal tissue or pathological benign",
                        "Usual Ductal Hyperplasia ",
                        "Flat Epithelia Atypia ",
                        "Atypical Ductal Hyperplasia",
                        "Ductal Carcinoma in Situ",
                        "Invasive Carcinoma"
                        ]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.df)

    def crop_or_pad_image(self, im):
        desired_size = self.crop_size
        old_size = im.size

        if old_size[0] > desired_size or old_size[1] > desired_size:
            # Crop the image
            left = (old_size[0] - desired_size) / 2
            top = (old_size[1] - desired_size) / 2
            right = (old_size[0] + desired_size) / 2
            bottom = (old_size[1] + desired_size) / 2

            im = im.crop((left, top, right, bottom))
        else:
            # Pad the image
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(im, ((desired_size - old_size[0]) // 2, (desired_size - old_size[1]) // 2))
            im = new_im

        return im

    def __getitem__(self, index):
        while True:
            try:
                item = self.df.iloc[index]
                image_path = os.path.join(self.root, item.image_path)
                image = Image.open(image_path).convert("RGB")
                image = self.crop_or_pad_image(image)
                break
            except:
                print(f"{item.image_path} failed.")
                os.system(f"rm -rf {item.image_path}")
                index = random.randint(0, self.__len__() - 1)

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[item.class_name]

        return image, label


class BRACS3ClsDataset(BRACS6ClsDataset):
    """
    BRACS dataset ROIs are scanned in 40x; and the image size is around 1800*1800 - 2400*2400;
    we get 224*224 at 5x; thus, the crop_size = 224*8.
    """
    def __init__(self, root, transform=None, train=True, is_retrieval=False):
        super(BRACS3ClsDataset, self).__init__(root, transform, train, is_retrieval)

        self.labels_dict = {"0_N": 0,
                            "1_PB": 0,
                            "2_UDH": 0,
                            "3_FEA": 1,
                            "4_ADH": 1,
                            "5_DCIS": 2,
                            "6_IC": 2
                            }

        self.classes = ["Non-cancerous	",
                        "Pre-cancerous	 ",
                        "Cancerous "
                        ]

from transformers import XLMRobertaTokenizer
BEIT3_Token = XLMRobertaTokenizer(
    "/mnt/sdd/vl_bertpath/3_contrastive_finetuning/beit3/beit3.spm"
    )

class PubmedDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):

        self.transform = transform
        self.root = root

        items = []
        index_file = os.path.join(root, "pubmed_set_retrieval.test.jsonl")
        with open(index_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                items.append(data)
            print("Load %d image-text pairs from %s. " % (len(items), index_file))
        self.items = items
    
    def __len__(self):
        return len(self.items)

    def _get_image(self, image_path):
        image = Image.open(image_path).convert("RGB")

        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for beit3 model
        else:
            image = self.transform(image)

        return image

    def __getitem__(self, idx):
        item = self.items[idx]
        image = self._get_image(item['image_path'])
        text = BEIT3_Token.decode(item['text_segment'])
        return image, text


class BookSetDataset(PubmedDataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.root = root

        items = []
        index_file = os.path.join(root, "books_set_retrieval.test.jsonl")
        with open(index_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                items.append(data)
            print("Load %d image-text pairs from %s. " % (len(items), index_file))
        self.items = items
