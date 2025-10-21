#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:
# MODEL_VERSION=vicuna-v1-3-7b
MODEL_VERSION=lmsys/vicuna-7b-v1.3

# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain

# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b"
########## DO NOT CHANGE ###########

GRAPH_TOWER="gpm"
if [ "$GRAPH_TOWER" == "vqvae2" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/vqvae.pth"
elif [ "$GRAPH_TOWER" == "hvqvae2" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/hvqvae.pth"
elif [ "$GRAPH_TOWER" == "gpm" ]; then
    INIT_CHECKPOINT_GNN="none"
else
    echo "Not supported graph tower"
fi

CHECKPOINT_FOLDER_PREFIX="./checkpoints/Graph-LLaVA-4C-${GRAPH_TOWER}-5ep-hlinear-fgprompt-neg-extend"
DATA_PATH="./data/PubChemSTM_data" # Path to the PubChem dataset

mkdir -p $CHECKPOINT_FOLDER_PREFIX

# Build output and log paths (MODEL_VERSION may contain a slash)
OUTPUT_DIR="$CHECKPOINT_FOLDER_PREFIX/llava-$GRAPH_TOWER-$MODEL_VERSION-pretrain"
LOG_PATH="$CHECKPOINT_FOLDER_PREFIX/llava-$GRAPH_TOWER-$MODEL_VERSION-pretrain_training.log"

# Ensure nested directories exist for both output and logs
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_PATH")"

PYTHONPATH="$PWD:$PYTHONPATH" deepspeed llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --graph_tower $GRAPH_TOWER \
    --graph_pooling "mean" \
    --init_checkpoint $INIT_CHECKPOINT_GNN \
    --tune_mm_mlp_adapter True \
    --mm_projector_type 'hlinear' \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --fp16 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 8000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --add_selfies False --motif_augmented True --add_lap_pe True --use_graph_rep False --extend_fg_prompt True  > "$LOG_PATH" &

