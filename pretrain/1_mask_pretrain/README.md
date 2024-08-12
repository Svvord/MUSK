## Masked Image/Text Pretraining

This code demonstrates how to train the MUSK model using unimodal image and text data, leveraging the unified mask modeling loss.


1. **Configure the Settings**
    - Set the paths for the preprocessed image and text files by assigning them to `--image_dir` and `--text_dir` in the `./configs/pretrain_musk_large.yaml` file. We provide example data [here](https://drive.google.com/drive/folders/1gaBMTnF4zVxt1hUn9qaZVsbXJeDp_-TH?usp=sharing).
    - Download the required text tokenizer `--tokenizer` [link](https://drive.google.com/file/d/1NJGch0cIhYzSSqTCJCRaCgJqDIG12d8H/view?usp=sharing) and specify its path.
    - Download the required image tokenizer `--tokenizer_weight` [link](https://drive.google.com/file/d/1fVxFnIPVZirEdg9tQ2vfv7MfEBOX9FuE/view?usp=sharing) and specify its path.
  
2. **Run the Pretraining Script**
    - Execute the pretraining script `./scripts/run_pathology_pretrain.sh` to train the MUSK model. The script allows for adjustment of the number of GPUs based on your available resources. By default, the script is configured for one node with 8 GPUs.

Monitor the log for `accuracy_mim` and `accuracy_mlm`. These metrics should increase steadily, indicating that the model is learning to recover the masked image tokens and text tokens.
