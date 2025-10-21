#!/bin/bash

# Evaluation script for property prediction using GPM model

# Activate conda environment
source /mnt/data1/gtoken/miniconda3/etc/profile.d/conda.sh
conda activate env_hight_ayush

# Configuration
MODEL_PATH="./checkpoints/Graph-LLaVA-4C-gpm-5ep-hlinear-fgprompt-neg-extend/graph-text-molgen/property_pred-llava-gpm-lmsys/vicuna-7b-v1.3-finetune_lora-5ep"
IN_FILE="./data/Molecule-oriented_Instructions/property_prediction-test.json"
ANSWERS_FILE="./eval_result/gpm-property_pred-5ep-test.json"
GRAPH_CHECKPOINT_PATH="none"
MODEL_BASE="lmsys/vicuna-7b-v1.3"

# Create eval_result directory if it doesn't exist
mkdir -p ./eval_result

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Check if input file exists
if [ ! -f "$IN_FILE" ]; then
    echo "Error: Input file does not exist: $IN_FILE"
    exit 1
fi

echo "Starting property prediction evaluation..."
echo "Model: $MODEL_PATH"
echo "Input: $IN_FILE"
echo "Output: $ANSWERS_FILE"

# Run property prediction sample generation
echo "Running property prediction sample generation..."
python -m llava.eval.molecule_metrics.generate_sample \
    --task property_pred \
    --model-path "$MODEL_PATH" \
    --in-file "$IN_FILE" \
    --answers-file "$ANSWERS_FILE" \
    --graph-checkpoint-path "$GRAPH_CHECKPOINT_PATH" \
    --model-base "$MODEL_BASE" \
    --conv-mode llava_v1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --repetition_penalty 1.0 \
    --num_beams 1 \
    --max-new-tokens 50 \
    --batch_size 1 \
    --motif_augmented

if [ $? -ne 0 ]; then
    echo "Error: Sample generation failed"
    exit 1
fi

echo "Sample generation completed successfully!"

# Run property prediction metrics evaluation
echo "Running property prediction metrics evaluation..."
python -m llava.eval.molecule_metrics.property_metrics \
    --eval_result_file "./eval_result/gpm-property_pred-5ep.json"

if [ $? -ne 0 ]; then
    echo "Error: Metrics evaluation failed"
    exit 1
fi

echo "Evaluation completed successfully!"
