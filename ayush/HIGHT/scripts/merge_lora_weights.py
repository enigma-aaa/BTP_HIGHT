import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import MM_ENCODER_CFG, get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    # graph encoder config
    mm_encoder_cfg = MM_ENCODER_CFG(init_checkpoint=args.graph_checkpoint_path)
    mm_encoder_cfg = mm_encoder_cfg.dict()
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu', mm_encoder_cfg=mm_encoder_cfg)

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/Graph-LLaVA-graphmvp-4C/llava-moleculestm-vicuna-v1-3-7b-pretrain/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora/MoleculeNet-llava-graphmvp-vicuna-v1-3-7b-finetune_lora-full/")
    parser.add_argument("--model-base", type=str, default="checkpoints/vicuna-v1-3-7b")
    parser.add_argument("--save-model-path", type=str, default="./checkpoints/Graph-LLaVA-graphmvp-4C/llava-moleculestm-vicuna-v1-3-7b-pretrain/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora/MoleculeNet-llava-graphmvp-vicuna-v1-3-7b-finetune_lora-full-merged/")
    parser.add_argument("--graph-checkpoint-path", type=str, default="checkpoints/MoleculeSTM/pretrained_GraphMVP/GraphMVP_G/model.pth")
    
    args = parser.parse_args()

    merge_lora(args)
