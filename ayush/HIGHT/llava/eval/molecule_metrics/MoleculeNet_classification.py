import argparse
import torch
import os
import pickle
from tqdm import tqdm
from typing import Generator, Dict, Any, Optional
import selfies
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, MM_ENCODER_CFG
from llava.datasets.smiles2graph import smiles2graph
# from llava.datasets.collators import _convert_dict_to_Data

def add_node_attr(
        data: Data,
        value: Any,
        attr_name: Optional[str] = None,
    ) -> Data:
        # TODO Move to `BaseTransform`.
        if attr_name is None:
            if data.x is not None:
                x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
                data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
            else:
                data.x = value
        else:
            data[attr_name] = value

        return data
SUPPORT_DATASETS = ["bace", "bbbp", "hiv"]
def _convert_dict_to_Data( data_dict: Dict) -> Data:
    data = Data(
        x=torch.asarray(data_dict['node_feat']),
        edge_attr=torch.asarray(data_dict['edge_feat']),
        edge_index=torch.asarray(data_dict['edge_index']),
    )
    if "lap_pe" in data_dict.keys():
        data = add_node_attr(data, data_dict["lap_pe"], attr_name="lap_pe")
    if "num_part" in data_dict.keys():
        data = add_node_attr(data, data_dict["num_part"], attr_name="num_part")
        # print(data.num_part)
    return data
    
def selfies2smiles(selfies_str):
    try:
        smiles_str = selfies.decoder(selfies_str)
    except:
        smiles_str = None
    return smiles_str

def smiles2selfies(smiles_str):
    try:
        selfies_str = selfies.encoder(smiles_str)
    except:
        selfies_str = None
    return selfies_str

def convert_label_to_int(label):
    label = label.strip()
    if label.lower() in ['active', "yes", "true"]:
        return 1
    elif label.lower() in ['inactive', "no", "false"]:
        return 0
    else:
        print("Unknown label:", label)
        return 0

def iterate_test_files(
    args, 
    batch_size:int=4,
)->Generator:
    if args.split == "random":
        in_file = os.path.join(args.dataspace, args.dataset, "processed", "instruct-random-test.pkl")
    else:
        in_file = os.path.join(args.dataspace, args.dataset, "processed", "instruct-test.pkl")
    with open(in_file, "rb") as f:
        list_data_dict = pickle.load(f)
        
        batch = []
        for raw in list_data_dict:
            instruction = raw['instruction']
            if args.add_selfies:
                selfies_str = smiles2selfies(raw['SMILES'])
                instruction += f" The compound SELFIES sequence is: {selfies_str}"
            graph = raw['graph']
            if args.motif_augmented:
                try:
                    graph = smiles2graph(raw['SMILES'],motif=args.motif_augmented)
                except Exception as e:
                    print(f"⚠️ Error processing SMILES: {raw['SMILES']}. Skipping this sample. Error: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
                    continue  # Skip this sample if there's an error

            batch.append((instruction, graph, raw['label']))
            if len(batch) == batch_size:
                yield zip(*batch)
                batch = []
        if len(batch) > 0:
            # Handle the last batch which may be smaller than batch_size
            yield zip(*batch) 



def _length_test_file(args)->int:
    in_file = os.path.join(args.dataspace, args.dataset, "processed", "instruct-random-test.pkl")
    with open(in_file, "rb") as f:
        list_data_dict = pickle.load(f)
        return len(list_data_dict)


def main(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    disable_torch_init()
    model_name = "llava_" + get_model_name_from_path(args.model_path)
    # graph encoder config
    mm_encoder_cfg = MM_ENCODER_CFG(init_checkpoint=args.graph_checkpoint_path)
    mm_encoder_cfg = mm_encoder_cfg.dict()
    tokenizer, model, _, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, mm_encoder_cfg=mm_encoder_cfg)
    # model.get_graph_tower().to(dtype=torch.float)
    model.get_graph_tower()._load_state_dict(args.graph_checkpoint_path,strict=False)
    model.get_graph_tower().requires_grad_(False)
    # model = model.to(torch.bfloat16)
    # Sampling 
    batch_size = args.batch_size
    outs = []
    if 'llama-2' in args.model_base:
        tokenizer.pad_token = tokenizer.eos_token
        args.conv_mode= "llava_llama_2"
    samples = 0
    for instructions, graphs, gts in tqdm(
        iterate_test_files(args, batch_size=batch_size), total=_length_test_file(args)//batch_size,
    ):  
        bs = len(instructions)
        graph_tensors = [_convert_dict_to_Data(graph).to(device) for graph in graphs]
        
        input_ids_batch = []
        stopping_criteria_batch = []
        for i in range(bs):
            qs = instructions[i]
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
            input_ids_batch.append(input_ids.squeeze(0))
            stopping_criteria_batch.append(stopping_criteria)
        # pad input_ids
        input_ids_batch = torch.nn.utils.rnn.pad_sequence(
            input_ids_batch,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )

        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         input_ids_batch,
        #         graphs=graph_tensors,
        #         do_sample=True,
        #         temperature=args.temperature,
        #         top_p=args.top_p,
        #         num_beams=args.num_beams,
        #         max_new_tokens=args.max_new_tokens,
        #         repetition_penalty=args.repetition_penalty,
        #         use_cache=True,
        #         stopping_criteria=stopping_criteria_batch
        #     )

        with torch.inference_mode():
            try:
                output_ids = model.generate(
                    input_ids_batch,
                    graphs=graph_tensors,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    use_cache=True,
                    stopping_criteria=stopping_criteria_batch
                )
            except Exception as e:
                print(f"⚠️ Skipping batch due to error: {e}")
                continue

        outputs = [] # list of str
        for i in range(bs):
            output = tokenizer.decode(output_ids[i, input_ids.shape[1]:]).strip()
            if output.endswith(stop_str):
                output = output[: -len(stop_str)]
            output = output.strip("</s>")
            output = output.strip()
            outputs.append(output)
        
        for instruction, gt, output in zip(instructions, gts, outputs):
            outs.append(
                {
                    "prompt": instruction,
                    "gt": gt,
                    "pred": output,
                }
            )
            if args.debug:
                print("\n", {"gt": gt, "outputs": output}, "\n")
        samples += bs
        # if samples > 20:
        #     break
    
    # compute metrics (ROC-AUC)
    preds = [convert_label_to_int(out["pred"]) for out in outs]
    gts = [convert_label_to_int(out["gt"]) for out in outs]
    print("ROC-AUC:", roc_auc_score(gts, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # /apdcephfs/private_yandrewchen_qy/GraphGPT/InstructMol/checkpoints/Graph-LLaVA/llava-moleculestm-vicuna-v1-3-7b-pretrain/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora-type2/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora-full/checkpoint-2234/config.json
    # ./checkpoints/Graph-LLaVA-moleculestm/llava-moleculestm-vicuna-v1-3-7b-pretrain/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora-type2/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora-full
    # parser.add_argument("--model-path", type=str, default="./checkpoints/Graph-LLaVA-4C/llava-moleculestm-vicuna-v1-3-7b-pretrain/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora-ep10/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora-full")
    # /apdcephfs/private_yandrewchen_qy/GraphGPT/InstructMol/checkpoints/Graph-LLaVA-moleculestm-molca/llava-moleculestm-vicuna-v1-3-7b-pretrain/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora-type2/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora-full/checkpoint-17872
    # parser.add_argument("--model-path", type=str, default="./checkpoints/Graph-LLaVA-moleculestm-molca/llava-moleculestm-vicuna-v1-3-7b-pretrain/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora-type2/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora-full")
    # InstructMol/checkpoints/Graph-LLaVA-graphmvp/llava-graphmvp-vicuna-v1-3-7b-pretrain-type2/MoleculeNet-llava-graphmvp-vicuna-v1-3-7b-finetune_lora-full
    # parser.add_argument("--model-path", type=str, default="./checkpoints/Graph-LLaVA-graphmvp-4C/llava-moleculestm-vicuna-v1-3-7b-pretrain/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora-ep10/MoleculeNet-llava-graphmvp-vicuna-v1-3-7b-finetune_lora-full")
    # InstructMol/checkpoints/Graph-LLaVA-graphmvp-4C-originGNN/llava-moleculestm-vicuna-v1-3-7b-pretrain/MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora-lr8e-5-orignGNN/MoleculeNet-llava-graphmvp-vicuna-v1-3-7b-finetune_lora-full/config.json
    # parser.add_argument("--model-path",type=str,default="/apdcephfs_qy3/share_1594716/yandrewchen/InstructMol/checkpoints/Graph-LLaVA-4C-vqvae-origdata/llava-vqvae-vicuna-v1-3-7b-pretrain/MoleculeNet-llava-vqvae-vicuna-v1-3-7b-finetune_lora/")
    # Graph-LLaVA-4C-moleculestm-origdata/llava-moleculestm-vicuna-v1-3-7b-pretrain/config.json
    parser.add_argument("--model-path", type=str, default="/mnt/data1/gtoken/ayush/HIGHT/checkpoints/Graph-LLaVA-4C-hvqvae2-5ep-hlinear-fgprompt-neg-extend/MoleculeNet-llava-hvqvae2-lmsys/vicuna-7b-v1.3-finetune_lora-large")
    parser.add_argument("--dataspace", type=str, default="/mnt/data1/gtoken/ayush/HIGHT/data/MoleculeNet_data/")
    parser.add_argument("--dataset", type=str, choices=SUPPORT_DATASETS, default="bbbp")
    # parser.add_argument("--graph-checkpoint-path", type=str, default="../drugchat/Mole-BERT/model_gin/Mole-BERT.pth")
    # parser.add_argument("--graph-checkpoint-path", type=str, default="../drugchat/Mole-BERT/vqencoder_zinc_standard_agent_epoch_60.pth")
    parser.add_argument("--graph-checkpoint-path", type=str, default="/mnt/data1/gtoken/ayush/HIGHT/checkpoints/hvqvae.pth")
    # parser.add_argument("--graph-checkpoint-path", type=str, default="checkpoints/MoleculeSTM/pretrained_GraphMVP/GraphMVP_G/model.pth")
    parser.add_argument("--model-base", type=str, default="lmsys/vicuna-7b-v1.3")

    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--split", type=str, default="random")
    parser.add_argument("--add_selfies", action="store_true")
    parser.add_argument("--motif_augmented", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
