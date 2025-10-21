#!/bin/bash

# Enhanced Pretrain Script with Detailed Timing Analysis (Fixed Version)
# This script adds comprehensive timing instrumentation to track each phase of training

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
TIMING_LOG_PATH="$CHECKPOINT_FOLDER_PREFIX/llava-$GRAPH_TOWER-$MODEL_VERSION-pretrain_timing.log"

# Ensure nested directories exist for both output and logs
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_PATH")"

# Function to log timing information
log_timing() {
    local phase="$1"
    local start_time="$2"
    local end_time="$3"
    local duration="$4"
    local details="$5"
    
    echo "[TIMING] $phase: Start=$start_time, End=$end_time, Duration=${duration}s, Details=$details" >> "$TIMING_LOG_PATH"
}

# Function to get current timestamp
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Function to calculate duration
calculate_duration() {
    local start="$1"
    local end="$2"
    local start_epoch=$(date -d "$start" +%s)
    local end_epoch=$(date -d "$end" +%s)
    echo $((end_epoch - start_epoch))
}

# Start timing
SCRIPT_START_TIME=$(get_timestamp)
echo "Starting pretraining at: $SCRIPT_START_TIME"
echo "Starting pretraining at: $SCRIPT_START_TIME" > "$TIMING_LOG_PATH"

# Phase 1: Environment Setup
echo "[TIMING] Phase 1: Environment Setup"
ENV_START=$(get_timestamp)
echo "Environment setup started at: $ENV_START"

# Check GPU availability
GPU_START=$(get_timestamp)
nvidia-smi >> "$TIMING_LOG_PATH" 2>&1
GPU_END=$(get_timestamp)
GPU_DURATION=$(calculate_duration "$GPU_START" "$GPU_END")
log_timing "GPU_CHECK" "$GPU_START" "$GPU_END" "$GPU_DURATION" "nvidia-smi check"

# Check Python environment
PYTHON_START=$(get_timestamp)
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')" >> "$TIMING_LOG_PATH" 2>&1
PYTHON_END=$(get_timestamp)
PYTHON_DURATION=$(calculate_duration "$PYTHON_START" "$PYTHON_END")
log_timing "PYTHON_ENV_CHECK" "$PYTHON_START" "$PYTHON_END" "$PYTHON_DURATION" "Python environment verification"

ENV_END=$(get_timestamp)
ENV_DURATION=$(calculate_duration "$ENV_START" "$ENV_END")
log_timing "ENVIRONMENT_SETUP" "$ENV_START" "$ENV_END" "$ENV_DURATION" "Complete environment setup"

# Phase 2: Data Loading
echo "[TIMING] Phase 2: Data Loading"
DATA_START=$(get_timestamp)
echo "Data loading started at: $DATA_START"

# Check data availability
DATA_CHECK_START=$(get_timestamp)
if [ -f "$DATA_PATH/hi_data_dict_lap.pkl" ]; then
    DATA_SIZE=$(du -h "$DATA_PATH/hi_data_dict_lap.pkl" | cut -f1)
    echo "Data file found: $DATA_PATH/hi_data_dict_lap.pkl (Size: $DATA_SIZE)"
    echo "Data file found: $DATA_PATH/hi_data_dict_lap.pkl (Size: $DATA_SIZE)" >> "$TIMING_LOG_PATH"
else
    echo "ERROR: Data file not found at $DATA_PATH/hi_data_dict_lap.pkl"
    echo "ERROR: Data file not found at $DATA_PATH/hi_data_dict_lap.pkl" >> "$TIMING_LOG_PATH"
    exit 1
fi
DATA_CHECK_END=$(get_timestamp)
DATA_CHECK_DURATION=$(calculate_duration "$DATA_CHECK_START" "$DATA_CHECK_END")
log_timing "DATA_AVAILABILITY_CHECK" "$DATA_CHECK_START" "$DATA_CHECK_END" "$DATA_CHECK_DURATION" "Data file verification"

DATA_END=$(get_timestamp)
DATA_DURATION=$(calculate_duration "$DATA_START" "$DATA_END")
log_timing "DATA_LOADING_PHASE" "$DATA_START" "$DATA_END" "$DATA_DURATION" "Complete data loading phase"

# Phase 3: Model Training
echo "[TIMING] Phase 3: Model Training"
TRAINING_START=$(get_timestamp)
echo "Training started at: $TRAINING_START"

# Execute training command
TRAINING_CMD_START=$(get_timestamp)
echo "Executing training command at: $TRAINING_CMD_START"

# Run the training directly
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
    --add_selfies False --motif_augmented True --add_lap_pe True --use_graph_rep False --extend_fg_prompt True \
    > "$LOG_PATH" 2>&1 &

TRAINING_PID=$!
echo "Training process started with PID: $TRAINING_PID"

# Monitor training progress
MONITOR_START=$(get_timestamp)
echo "Starting training monitoring at: $MONITOR_START"

# Function to monitor training
monitor_training() {
    local last_step=0
    local last_time=$(date +%s)
    local check_count=0
    
    while kill -0 $TRAINING_PID 2>/dev/null; do
        sleep 30  # Check every 30 seconds
        check_count=$((check_count + 1))
        
        # Check if log file has new content
        if [ -f "$LOG_PATH" ]; then
            # Extract latest step information
            latest_step=$(grep -o '[0-9]*%|' "$LOG_PATH" | tail -1 | grep -o '[0-9]*' | head -1)
            if [ ! -z "$latest_step" ] && [ "$latest_step" -gt "$last_step" ]; then
                current_time=$(date +%s)
                time_diff=$((current_time - last_time))
                echo "[TIMING] Training Progress: Step $latest_step, Time since last update: ${time_diff}s"
                echo "[TIMING] Training Progress: Step $latest_step, Time since last update: ${time_diff}s" >> "$TIMING_LOG_PATH"
                last_step=$latest_step
                last_time=$current_time
            fi
            
            # Log progress every 10 checks (5 minutes)
            if [ $((check_count % 10)) -eq 0 ]; then
                echo "[TIMING] Monitoring check #$check_count, Training still running..."
                echo "[TIMING] Monitoring check #$check_count, Training still running..." >> "$TIMING_LOG_PATH"
            fi
        fi
    done
    
    echo "[TIMING] Training process ended, monitoring stopped after $check_count checks"
    echo "[TIMING] Training process ended, monitoring stopped after $check_count checks" >> "$TIMING_LOG_PATH"
}

# Start monitoring in background
monitor_training &
MONITOR_PID=$!

# Wait for training to complete
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

TRAINING_END=$(get_timestamp)
TRAINING_DURATION=$(calculate_duration "$TRAINING_START" "$TRAINING_END")
log_timing "TRAINING_PHASE" "$TRAINING_START" "$TRAINING_END" "$TRAINING_DURATION" "Complete training phase, Exit code: $TRAINING_EXIT_CODE"

# Phase 4: Post-training Analysis
echo "[TIMING] Phase 4: Post-training Analysis"
POST_START=$(get_timestamp)

# Generate timing analysis
if [ -f "$LOG_PATH" ]; then
    echo "Generating timing analysis..."
    mkdir -p "$CHECKPOINT_FOLDER_PREFIX/timing_analysis"
    python3 scripts/timing_analysis.py "$LOG_PATH" --output-dir "$CHECKPOINT_FOLDER_PREFIX/timing_analysis"
    ANALYSIS_END=$(get_timestamp)
    ANALYSIS_DURATION=$(calculate_duration "$POST_START" "$ANALYSIS_END")
    log_timing "TIMING_ANALYSIS" "$POST_START" "$ANALYSIS_END" "$ANALYSIS_DURATION" "Generated timing analysis report"
fi

POST_END=$(get_timestamp)
POST_DURATION=$(calculate_duration "$POST_START" "$POST_END")
log_timing "POST_TRAINING_ANALYSIS" "$POST_START" "$POST_END" "$POST_DURATION" "Complete post-training analysis"

# Final summary
SCRIPT_END_TIME=$(get_timestamp)
TOTAL_DURATION=$(calculate_duration "$SCRIPT_START_TIME" "$SCRIPT_END_TIME")

echo ""
echo "=========================================="
echo "PRETRAINING COMPLETED"
echo "=========================================="
echo "Start Time: $SCRIPT_START_TIME"
echo "End Time: $SCRIPT_END_TIME"
echo "Total Duration: ${TOTAL_DURATION} seconds ($(($TOTAL_DURATION / 60)) minutes)"
echo "Training Exit Code: $TRAINING_EXIT_CODE"
echo "Log File: $LOG_PATH"
echo "Timing Log: $TIMING_LOG_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

log_timing "TOTAL_SCRIPT_EXECUTION" "$SCRIPT_START_TIME" "$SCRIPT_END_TIME" "$TOTAL_DURATION" "Complete script execution"

echo "Timing analysis complete. Check $TIMING_LOG_PATH for detailed timing information."
