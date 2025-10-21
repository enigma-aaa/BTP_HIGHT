#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:


################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="lmsys/vicuna-7b-v1.3" # Corrected to the base model ID
################## VICUNA ##################

# ################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
# ################## LLaMA-2 ##################

GRAPH_TOWER="gpm"
CHECKPOINT_FOLDER_PREFIX="./checkpoints/Graph-LLaVA-4C-${GRAPH_TOWER}-5ep-hlinear-fgprompt-neg-extend"

if [ "$GRAPH_TOWER" == "hvqvae2" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/hvqvae.pth"
    # ADAPTER should point to the mm_projector.bin from the pre-training output
    ADAPTER="${CHECKPOINT_FOLDER_PREFIX}/llava-hvqvae2-lmsys/vicuna-7b-v1.3-pretrain/mm_projector.bin"
elif [ "$GRAPH_TOWER" == "vqvae2" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/vqvae.pth"
    ADAPTER="${CHECKPOINT_FOLDER_PREFIX}/llava-hvqvae2-lmsys/vicuna-7b-v1.3-pretrain/mm_projector.bin"
elif [ "$GRAPH_TOWER" == "gpm" ]; then
    INIT_CHECKPOINT_GNN="none"
    PRETRAIN_OUTPUT="${CHECKPOINT_FOLDER_PREFIX}/llava-${GRAPH_TOWER}-${MODEL_VERSION}-pretrain"
    ADAPTER="${PRETRAIN_OUTPUT}/mm_projector.bin"
else
    echo "Not supported graph tower"
fi


# TASK="MoleculeNet"
# mkdir -p $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-large
# mkdir -p $(dirname $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-full-lr4e-5_training.log)
# deepspeed llava/train/train_mem.py \
#     --deepspeed scripts/zero2.json \
#     --lora_enable True \
#     --model_name_or_path $MODEL_VERSION \
#     --version $PROMPT_VERSION \
#     --data_path ./data/MoleculeNet_data/ \
#     --data_type $TASK \
#     --graph_tower $GRAPH_TOWER \
#     --init_checkpoint $INIT_CHECKPOINT_GNN \
#     --pretrain_mm_mlp_adapter $ADAPTER \
#     --mm_projector_type 'hlinear' \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --fp16 False \
#     --output_dir $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-large \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 20 \
#     --learning_rate 4e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --dataloader_num_workers 4 \
#     --report_to tensorboard \
#     --add_selfies False --motif_augmented True --add_lap_pe True --use_graph_rep False --extend_fg_prompt True  > $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-full-lr4e-5_training.log &
# wait


TASK="property_pred"
mkdir -p $CHECKPOINT_FOLDER_PREFIX/graph-text-molgen/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep
mkdir -p $(dirname $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep_training.log)
PYTHONPATH="$PWD:$PYTHONPATH" deepspeed llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./data/Molecule-oriented_Instructions/property_prediction-train.json \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --graph_pooling "mean" \
    --init_checkpoint $INIT_CHECKPOINT_GNN \
    --pretrain_mm_mlp_adapter $ADAPTER \
    --mm_projector_type 'hlinear' \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --fp16 False \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/graph-text-molgen/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to none \
    --add_selfies False --motif_augmented True --add_lap_pe True --use_graph_rep False --extend_fg_prompt True  > $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep_training.log &
wait


# TASK="forward_pred"
# mkdir -p $CHECKPOINT_FOLDER_PREFIX/graph-text-molgen/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep16bz
# mkdir -p $(dirname $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep_training.log)
# deepspeed llava/train/train_mem.py \
#     --deepspeed scripts/zero2.json \
#     --lora_enable True \
#     --model_name_or_path $MODEL_VERSION \
#     --version $PROMPT_VERSION \
#     --data_path ./data/Molecule-oriented_Instructions/forward_reaction_prediction_train.json \
#     --motif_augmented True \
#     --data_type forward_pred \
#     --graph_tower $GRAPH_TOWER \
#     --init_checkpoint $INIT_CHECKPOINT_GNN \
#     --pretrain_mm_mlp_adapter $ADAPTER \
#     --mm_projector_type 'hlinear' \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --fp16 False \
#     --output_dir $CHECKPOINT_FOLDER_PREFIX/graph-text-molgen/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep16bz \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --dataloader_num_workers 4 \
#     --report_to none \
#     --add_selfies False --motif_augmented True --add_lap_pe True --use_graph_rep False   > $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep_training.log &
# wait

# TASK="reagent_pred"
# mkdir -p $CHECKPOINT_FOLDER_PREFIX/graph-text-molgen/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep16bz
# mkdir -p $(dirname $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep_training.log)
# deepspeed llava/train/train_mem.py \
#     --deepspeed scripts/zero2.json \
#     --lora_enable True \
#     --model_name_or_path $MODEL_VERSION \
#     --version $PROMPT_VERSION \
#     --data_path ./data/Molecule-oriented_Instructions/reagent_prediction_train.json \
#     --motif_augmented True \
#     --data_type reagent_pred \
#     --graph_tower $GRAPH_TOWER \
#     --init_checkpoint $INIT_CHECKPOINT_GNN \
#     --pretrain_mm_mlp_adapter $ADAPTER \
#     --mm_projector_type 'hlinear' \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --fp16 False \
#     --output_dir $CHECKPOINT_FOLDER_PREFIX/graph-text-molgen/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep16bz \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 10 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --dataloader_num_workers 4 \
#     --report_to none \
#     --add_selfies False --motif_augmented True --add_lap_pe True --use_graph_rep False   > $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep_training.log &
# wait

# TASK="retrosynthesis"
# mkdir -p $CHECKPOINT_FOLDER_PREFIX/graph-text-molgen/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep16bz
# mkdir -p $(dirname $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep_training.log)
# deepspeed llava/train/train_mem.py \
#     --deepspeed scripts/zero2.json \
#     --lora_enable True \
#     --model_name_or_path $MODEL_VERSION \
#     --version $PROMPT_VERSION \
#     --data_path ./data/Molecule-oriented_Instructions/retrosynthesis_train.json \
#     --motif_augmented True \
#     --data_type retrosynthesis \
#     --graph_tower $GRAPH_TOWER \
#     --init_checkpoint $INIT_CHECKPOINT_GNN \
#     --pretrain_mm_mlp_adapter $ADAPTER \
#     --mm_projector_type 'hlinear' \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --fp16 False \
#     --output_dir $CHECKPOINT_FOLDER_PREFIX/graph-text-molgen/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep16bz \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --dataloader_num_workers 4 \
#     --report_to none \
#     --add_selfies False --motif_augmented True --add_lap_pe True --use_graph_rep False  > $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep_training.log &
# wait

# TASK="molcap"
# mkdir -p $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-50ep-t2
# mkdir -p $(dirname $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep_training.log)
# deepspeed llava/train/train_mem.py \
#     --deepspeed scripts/zero2.json \
#     --lora_enable True \
#     --model_name_or_path $MODEL_VERSION \
#     --version $PROMPT_VERSION \
#     --data_path ./data/ChEBI_data/train.txt \
#     --graph_tower $GRAPH_TOWER \
#     --use_graph_rep False \
#     --init_checkpoint $INIT_CHECKPOINT_GNN \
#     --pretrain_mm_mlp_adapter $ADAPTER \
#     --mm_projector_type 'hlinear' \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --fp16 False \
#     --output_dir $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-50ep-t2 \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 10 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 5 \
#     --learning_rate 8e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --dataloader_num_workers 4 \
#     --report_to none \
#     --add_selfies False --motif_augmented True --add_lap_pe True --use_graph_rep False > $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-5ep_training.log &
# wait