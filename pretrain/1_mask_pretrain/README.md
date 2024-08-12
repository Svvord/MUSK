## Masked Image/Text Pretraining

This code demonstrates how to train the MUSK model using unimodal image and text data, leveraging the unified mask modeling loss.


1. **Configure the Settings**
    - Set the paths for the preprocessed image and text files by assigning them to `--image_dir` and `--text_dir` in the `./configs/pretrain_musk_large.yaml` file.
    - Download the required `--tokenizer` and specify its path.
    - Download the image tokenizer and set its path using `--tokenizer_weight`.
  
2. **Run the Pretraining Script**
    - Execute the pretraining script `./scripts/run_pathology_pretrain.sh` to train the MUSK model. The script allows for adjustment of the number of GPUs based on your available resources. By default, the script is configured for one node with 8 GPUs.

Monitor the log for `accuracy_mim` and `accuracy_mlm`. These metrics should increase steadily, indicating that the model is learning to recover the masked image tokens and text tokens.