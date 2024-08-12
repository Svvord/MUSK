
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HOST_NUM=1
export INDEX=0
export CHIEF_IP=localhost
export HOST_GPU_NUM=4

BEIT3_DIR="/mnt/radonc-li01/private/xiangjx/code/musk_v2/3_contrastive_pretrain/beit3"
DATA_Path="/mnt/radonc-li01/public/quilt1m"
SAVE_DIR="large_384_quilt1m_decoder_loss_v1"


torchrun --nproc_per_node=${HOST_GPU_NUM} --nnodes=${HOST_NUM} --node_rank=${INDEX} \
      --master_addr=${CHIEF_IP} --master_port=1639 ../run.py \
       --model musk_large_patch16_384 \
       --input_size 384 \
       --task quilt1m_pathcap \
       --batch_size 4 \
       --num_workers 8 \
       --layer_decay 0.85 \
       --lr 1e-4 \
       --opt_betas 0.9 0.95 \
       --epochs 20 \
       --warmup_epochs 2 \
       --drop_path 0.2 \
       --sentencepiece_model ${BEIT3_DIR}/beit3.spm \
       --finetune ${BEIT3_DIR}/beit3_large_itc_patch16_224.zip  \
       --data_path ${DATA_Path} \
       --output_dir ./results/${SAVE_DIR} \
       --log_dir ./results/${SAVE_DIR}/log \
       --weight_decay 0.05 \
       --seed 42 \
       --save_ckpt_freq 5 \
       --enable_deepspeed \
       --text_mask_prob 0.15 \
       --checkpoint_activations