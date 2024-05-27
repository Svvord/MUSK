#!/bin/bash


# >>>>>>>>>>> zero-shot cross-modal retrieval >>>>>>>>>>> #
 python3 -m clip_benchmark.cli eval --pretrained_model models.txt \
        --dataset   "bookset_retrieval"  \
        --task "zeroshot_retrieval" \
        --batch_size 512 \
        --num_workers 16 \
        --seed 42 \
        --recall_k 1 10 50 \
        --dataset_root "" \
        --output "./results/benchmark_mm_retrieval.json"

seed_list=(123 456 789 321 654)  # Add your seeds here
shot_list=(1 2 4 8 10)  # change the number of shot

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
          --dataset_root "" \
          --output "./results/benchmark_fs_${k_shot}shot_seed${seed}.json"
  done
done

seed_list=(123)  # Add your seeds here
# # >>>>>>>>>>> linear probe >>>>>>>>>>> #
for seed in "${seed_list[@]}"
do
    python3 -m clip_benchmark.cli eval --pretrained_model models.txt \
        --dataset  "nct_crc" "pcam" "skin" "sicap" "pannuke" "unitopatho" "wsss4luad" "osteo" "lc25" "renal_cell" "bracs6cls" "bracs3cls" \
        --task "linear_probe" \
        --batch_size 1024 \
        --num_workers 8 \
        --fewshot_k -1 \
        --seed $seed \
        --dataset_root "" \
        --output "./results/linear_probe_${seed}.json"
done


# # >>>>>>>>>>> image retrieval >>>>>>>>>>> #
python3 -m clip_benchmark.cli eval --pretrained_model models.txt \
        --dataset   "unitopatho_retrieval" "bracs_retrieval" \
        --task "image_retrieval" \
        --batch_size 512 \
        --num_workers 16 \
        --seed 41 \
        --dataset_root "" \
        --output "./results/benchmark_image_retrieval.json"


# # >>>>>>>>>>> zero-shot cls >>>>>>>>>>> #
python3 -m clip_benchmark.cli eval --pretrained_model models.txt \
    --dataset "nct_crc" "pcam" "skin" "sicap" "pannuke" "unitopatho" "wsss4luad" "osteo" "lc25" "renal_cell" "bracs6cls" "bracs3cls" \
    --task "zeroshot_classification" \
    --batch_size 512 \
    --dataset_root "" \
    --output "./results/benchmark_zeroshot_cls.json"




# ------ full list -------- #
# ViT-B-32,/mnt/sdd/models/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K/open_clip_pytorch_model.bin
# PLIP,/mnt/sdd/models/plip
# ViT-B-32,/mnt/sdd/models/QuiltNet-B-32/open_clip_pytorch_model.bin
# ViT-B-16,/mnt/sdd/models/CLIP-ViT-B-16-laion2B-s34B-b88K/open_clip_pytorch_model.bin
# ViT-B-16,/mnt/sdd/models/QuiltNet-B-16/open_clip_pytorch_model.bin
# biomedclip,/mnt/sdd/models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

# TransPath,/mnt/sdd/vl_bertpath/4_evaluate_benchmark/vision_tasks/TransPath/ctranspath.pth
# beit3_base_patch16_224_retrieval,/mnt/sdd/vl_bertpath/3_contrastive_finetuning/scripts/results/bertpath_base_224/model.pth
# beit3_large_patch16_384_retrieval,/mnt/sdd/vl_bertpath/3_contrastive_finetuning/scripts/results/bertpath_large_384/model.pth
# conch,/mnt/sdd/models/conch/conch.pt

