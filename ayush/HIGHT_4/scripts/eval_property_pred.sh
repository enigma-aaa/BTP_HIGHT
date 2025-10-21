#!/bin/bash

# Evaluation script for property prediction tasks
# Based on commands from results/eval_commands.txt

echo "Starting property prediction evaluation..."

# Set up paths
CHECKPOINT_FOLDER_PREFIX="./checkpoints/Graph-LLaVA-4C-gpm-5ep-hlinear-fgprompt-neg-extend"
MODEL_PATH="${CHECKPOINT_FOLDER_PREFIX}/graph-text-molgen/property_pred-llava-gpm-lmsys/vicuna-7b-v1.3-finetune_lora-5ep"
IN_FILE="./data/Molecule-oriented_Instructions/property_prediction-test.json"
ANSWERS_FILE="./eval_result/gpm-property_pred-5ep.json"
GRAPH_CHECKPOINT_PATH="none"  # GPM doesn't need a checkpoint
MODEL_BASE="lmsys/vicuna-7b-v1.3"

# Create output directory
mkdir -p ./eval_result

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path $MODEL_PATH does not exist!"
    echo "Please make sure you have completed training and the model is saved at the expected location."
    exit 1
fi

# Check if input file exists
if [ ! -f "$IN_FILE" ]; then
    echo "Error: Input file $IN_FILE does not exist!"
    exit 1
fi

echo "Running property prediction sample generation..."

# Generate samples for property prediction
python -m llava.eval.molecule_metrics.generate_sample \
    --task property_pred \
    --model-path "${MODEL_PATH}" \
    --in-file "${IN_FILE}" \
    --answers-file "${ANSWERS_FILE}" \
    --graph-checkpoint-path "${GRAPH_CHECKPOINT_PATH}" \
    --model-base "${MODEL_BASE}" \
    --batch_size 1 --temperature 0.2 --top_p 1.0 \
    --debug \
    --motif_augmented

echo "Running property prediction metrics evaluation..."

# Evaluate property prediction metrics
conda run -n env_hight_ayush python -m llava.eval.molecule_metrics.property_metrics --eval_result_file "${ANSWERS_FILE}"