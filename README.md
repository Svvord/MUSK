MUSK  
===========
## A Vision-Language Foundation Model for Precision Oncology

<img src="MUSK.png" width="300px" align="right" />

**Abstract:** Clinical decision-making is a process driven by multimodal data, including clinical notes and pathologic characteristics. Artificial intelligence approaches that can effectively integrate multimodal data hold significant promise to advance clinical care. However, the scarcity of well-annotated multimodal datasets in the clinical setting hinders the development of useful models.  Here, we develop **M**ultimodal transformer with **U**nified ma**SK** modeling (MUSK), a vision-language foundation model designed to leverage large-scale, unlabeled, unpaired image-text data. MUSK is pre-trained on 50 million pathology images and 1 billion pathology-related text tokens using unified masked modeling. MUSK is further pre-trained on 1 million pathology image-text pairs to align vision and language features efficiently. After pretaining, MUSK is applied to a wide range of downstream tasks involving pathology images and/or text with minimal or no further fine-tuning.  MUSK achieves superior performance across 19 patch-level and slide-level benchmarks, including image-to-text and text-to-image retrieval, visual question answering, and image classification. Importantly, MUSK shows promising performance in outcome prediction, including melanoma relapse prediction, pan-cancer prognosis prediction, and immunotherapy response prediction in lung and gastro-esophageal cancers. MUSK effectively combines complementary information from pathology images and clinical reports and can potentially improve diagnosis and precision cancer therapy.


## Pipeline

_MUSK Pretraining_:
1. Preprocess WSI (extracting foreground, generating tiles, and saving results into tsv files).
2. Masked pretraining with unpair image patches (50 million) and text corpora (1 billion tokens).
3. Contrastive learning pretraining with image-text pairs (1 million).

_Downstream Evaluation_:
1. **Cancer Diagnosis/Detection**
   - image-text retrieval
   - Pathology Visual Question Answering
   - image classification (zero-shot, few-shot, linear probe)
   - image-to-image retrieval
2. **Outcome Prediction**
   - Melanoma relapse prediction
   - Pan-cancer prognosis prediction
   - Lung cancer immunotherapy response prediction
   - Gastric cancer immunotherapy response prediction


## Installation

First clone the repo and cd into the directory:
```shell
git clone https://github.com/lilab-stanford/MUSK
cd MUSK
```

Create a new enviroment with anaconda.
```shell
conda create -n musk python=3.10 -y
conda activate musk
pip install --upgrade pip
pip install -r requirements.txt
```

## Basic Usage: MUSK as a Vision-Language Encoder

Please refer to `./musk/demo.ipynb` for a demonstration. 

Download the [model weight](https://drive.google.com/file/d/1Suwo7xumPVeNW-_ggJGi0VkTmaEGsOGo/view?usp=sharing) and [tokenizer](https://drive.google.com/file/d/1NJGch0cIhYzSSqTCJCRaCgJqDIG12d8H/view?usp=sharing), and place them in the `./musk/models` directory.


```shell
cd ./musk
```

1. Load the MUSK model  

```python
from timm.models import create_model
model = create_model("musk_large_patch16_384", vocab_size=64010).eval()
utils.load_model_and_may_interpolate("./models/musk.pth", model, 'model|module', '')
model.to(DEVICE, dtype=torch.float16)
model.eval()
```

2. Encode image with MUSK 
```python
with torch.inference_mode():
   image_embeddings = model(
      image=img_tensor.to(DEVICE, dtype=torch.float16),
      with_head=True, 
      out_norm=True
      )[0]  # return (vision_cls, text_cls)
```

The `with_head` parameter controls the projection head at the last layer. Set this parameter to `True` when performing image-text retrieval. For tasks like image classification or multiple instance learning (MIL), you can disable it by setting it to `False`. The `out_norm` parameter handles output normalization and is enabled by default (`True`).



3. Encode text with MUSK
```python
tokenizer = XLMRobertaTokenizer("./models/tokenizer.spm")
text = 'histopathology image of lung adenocarcinoma'
txt_ids, pad = xlm_tokenizer(txt, tokenizer, max_len=100)

with torch.inference_mode():
   text_embeddings = model(
      text_description=txt_ids.to(DEVICE),
      padding_mask=pad.to(DEVICE),
      with_head=True, 
      out_norm=True
   )[1]  # return (vision_cls, text_cls)
```
Both `with_head` and `out_norm` should keep the same settings as those used in image encoding.

## Model Pretraining

Masked pretraining [instructions](https://github.com/lilab-stanford/MUSK/tree/main/pretrain/1_mask_pretrain).

Contrastive pretraining [instructions](https://github.com/lilab-stanford/MUSK/tree/main/pretrain/2_contrastive_pretrain).

## Evaluation on Cancer Diagnosis/Detection

Please refer to `./benchmarks/demo.ipynb` for a demonstration. 

This section reproduces the results of cancer diagnosis/detection benchmarks, including image-text retrieval, image classification, image-image retrieval, and more. The evaluation code is all-in-one which adapted from the [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark).

The evaluated dataset includes:
<small>
* BookSet: [Download Link](https://warwick.ac.uk/fac/cross_fac/tia/data/arch)
* PubmedSet: [Download Link](https://warwick.ac.uk/fac/cross_fac/tia/data/arch)
* PatchCamelyon: [Download Link](https://patchcamelyon.grand-challenge.org/)
* NCT-CRC-HE-100K: [Project Link](https://zenodo.org/record/1214456)
* SICAPv2: [Download Link](https://www.biicancer.com/sicapv2-dataset)
* Osteo: [Project Link](https://zenodo.org/record/3626656)
* RenalCell: [Download Link](https://www.kaggle.com/mnassrib/renal-cell-carcinoma-rcc)
* SkinCancer: [Project Link](https://www.isic-archive.com/)
* LC25000: [Download Link](https://www.kaggle.com/andrewmvd/lung-and-colon-cancer-histopathological-images)
* PanNuke: [Project Link](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)
* UniToPatho: [Project Link](https://zenodo.org/record/4295977)
* WSSS4LUAD: [Download Link](https://github.com/sjin010/WSSS4LUAD)
* BRACS (3 cls): [Download Link](https://www.bracs.icar.cnr.it/download/)
* BRACS (6 cls): [Download Link](https://www.bracs.icar.cnr.it/download/)
</small>

First, download the necessary datasets. For demonstrations, we provide the example datasets [here](https://drive.google.com/drive/folders/15xlGg3HE4rVWz7ofg1rsgNwHGIFiO6Qa?usp=sharing). Download and unzip it to a local path, for example `/root/user/data/downstreams_demo`, then, change the directory path `dataset_root=/root/user/data/downstreams_demo`. The code will automatically extract features and perform evaluations.


The main file is `clip_benchmark.cli` and includes the following options:
- `--pretrained_model`: Specifies the model name and the path to its weights.
- `--dataset`: Indicates the evaluation dataset(s); multiple datasets can be specified.
- `--dataset_root`: The root of datasets.
- `--task`: Defines the evaluation task.
- `--batch_size`: Sets the batch size for feature extraction.
- `--output`: Specifies where to save the output results.

Here are some example commands:


```shell
# >>>>>>>>>>> zero-shot cross-modal retrieval >>>>>>>>>>> #
 python3 -m clip_benchmark.cli eval --pretrained_model models.txt \
        --dataset   "bookset_retrieval"  \
        --task "zeroshot_retrieval" \
        --batch_size 512 \
        --num_workers 16 \
        --seed 42 \
        --recall_k 1 10 50 \
        --dataset_root "/root/user/data/downstreams_demo" \
        --output "./results/benchmark_mm_retrieval.json"
```


```shell
# >>>>>>>>>>> few-shot linear probe >>>>>>>>>>> #
for k_shot in "${shot_list[@]}"
do
  for seed in "${seed_list[@]}"
  do
      python3 -m clip_benchmark.cli eval --pretrained_model models.txt \
          --dataset  "nct_crc" "pcam" "skin" "sicap" "pannuke" "unitopatho" "wsss4luad" "osteo" "lc25" "renal_cell" "bracs6cls" "bracs3cls" \
          --task "linear_probe" \
          --batch_size 512 \
          --num_workers 16 \
          --fewshot_k $k_shot \
          --seed $seed \
          --dataset_root "/root/user/data/downstreams_demo" \
          --output "./results/benchmark_fs_${k_shot}shot_seed${seed}.json"
  done
done
```

```shell
# >>>>>>>>>>> image retrieval >>>>>>>>>>> #
python3 -m clip_benchmark.cli eval --pretrained_model models.txt \
        --dataset   "unitopatho_retrieval" "bracs_retrieval" \
        --task "image_retrieval" \
        --batch_size 512 \
        --num_workers 16 \
        --seed 41 \
        --dataset_root "/root/user/data/downstreams_demo" \
        --output "./results/image_retrieval/benchmark_image_retrieval.json"
```

and more tasks in `./benchmarks/demo.ipynb`.


## Evaluation on Outcome Prediction


This section reproduces the outcome prediction results presented in the paper. All evaluated tasks utilize a consistent training and evaluation pipeline:

1. **Preprocess the whole slide images** following the [CLAM](https://github.com/mahmoodlab/CLAM) method, which includes tissue segmentation and stitching.
2. **Extract image embeddings** using MUSK.
3. **Preprocess the pathology reports** using GPT-4 to convert them into structured reports, as explained in the paper.
4. **Extract report embeddings** using MUSK, with the structured reports (`./clinical/workspace/data/tcga_reports_gpt4.csv`) as inputs.
5. **Finetune a linear regression model** that combines both image and report embeddings as inputs.  Run `python3 train_mil.py` in `./clinical`.


Please refer to `./clinical/demo.ipynb` for demonstrations. 

To reproduce the results, download the processed features for `IDH (subtyping)`[link](https://drive.google.com/file/d/1XfKB1qSjjujmQ4VwAGN5D9G6jt8bfyS6/view?usp=sharing) and `TCGA (prognosis)`[link](https://drive.google.com/file/d/1MLLT-iMMNACwR9TzcO4MfzDiF6nRiDn8/view?usp=sharing) into the `./clinical/workspace/data`, and then run `python3 train_mil.py`. 

To switch tasks, modify the `fname` in `train_mil.py` to other configurations in the `configs` directory.



## Acknowledgements

The project was built on top of many open-source repositories such as [Quilt1M](https://github.com/wisdomikezogwo/quilt1m) (training data image-text pairs), [torchscale](https://github.com/microsoft/torchscale) (model implementation), [accelerate](https://github.com/huggingface/accelerate) (model pretraining), [deepspeed](https://github.com/microsoft/DeepSpeed) (model pretraining), [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) (downstream finetuning), and [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark) (model evaluation). We thank the authors and developers for their contributions.




## License

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the MUSK model and its derivatives, which include models trained on outputs from the MUSK model or datasets created from the MUSK model, is prohibited and requires prior approval.