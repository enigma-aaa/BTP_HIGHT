set -euo pipefail

GRAPH_TOWER="gpm"
MODEL_VERSION="lmsys/vicuna-7b-v1.3"
CHECKPOINT_FOLDER_PREFIX="./checkpoints/Graph-LLaVA-4C-${GRAPH_TOWER}-5ep-hlinear-fgprompt-neg-extend"
PRETRAIN_OUTPUT="${CHECKPOINT_FOLDER_PREFIX}/llava-${GRAPH_TOWER}-${MODEL_VERSION}-pretrain"
FINETUNE_OUTPUT="${CHECKPOINT_FOLDER_PREFIX}/graph-text-molgen/property_pred-llava-${GRAPH_TOWER}-${MODEL_VERSION}-finetune_lora-5ep"

# Activate environment and setup runtime
source ~/.bashrc >/dev/null 2>&1 || true
conda activate env_hight_ayush >/dev/null 2>&1 || source activate env_hight_ayush >/dev/null 2>&1 || true
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TRANSFORMERS_VERBOSITY=error

# GPM environment variables
export GPM_NUM_PATTERNS=${GPM_NUM_PATTERNS:-4}
export GPM_PATTERN_SIZE=${GPM_PATTERN_SIZE:-3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

mkdir -p ./eval_result

# Ensure projector present in finetuned folder (eval loader expects it there)
if [ -f "${PRETRAIN_OUTPUT}/mm_projector.bin" ] && [ ! -f "${FINETUNE_OUTPUT}/mm_projector.bin" ]; then
  cp -f "${PRETRAIN_OUTPUT}/mm_projector.bin" "${FINETUNE_OUTPUT}/mm_projector.bin"
fi

python -u -m llava.eval.molecule_metrics.generate_sample \
    --task property_pred \
    --model-path "$FINETUNE_OUTPUT" \
    --in-file ./data/Molecule-oriented_Instructions/property_prediction-test.json \
    --answers-file ./eval_result/gpm-property_pred-5ep.jsonl \
    --graph-checkpoint-path none \
    --model-base "$MODEL_VERSION" \
    --conv-mode v1 \
    --batch_size 1 --num_beams 1 --max-new-tokens 64 \
    --temperature 0.2 --top_p 1.0 \
    --debug \
    --add_selfies --motif_augmented

python -u -m llava.eval.molecule_metrics.property_metrics --eval_result_file ./eval_result/gpm-property_pred-5ep.jsonl