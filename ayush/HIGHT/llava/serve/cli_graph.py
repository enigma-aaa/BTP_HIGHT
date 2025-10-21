import argparse
import torch
from llava.model.language_model.llava_graph_llama import LlavaGraphLlamaConfig
from llava.model.multimodal_encoder.builder import build_graph_tower

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, MM_ENCODER_CFG
from llava.mol_utils import check_smiles_validity
from llava.datasets.smiles2graph import smiles2graph

from typing import Dict
from transformers import TextStreamer
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.eval.molecule_metrics.MoleculeNet_classification import _convert_dict_to_Data
# def _convert_dict_to_Data(data_dict: Dict) -> Data:
#     return Data(
#         x=torch.asarray(data_dict['node_feat']),
#         edge_attr=torch.asarray(data_dict['edge_feat']),
#         edge_index=torch.asarray(data_dict['edge_index']),
#     )

fgroup_names = ['amide',
 'isocyanate',
 'isothiocyanate',
 'containing',
 'nitro',
 'nitroso',
 'oximes',
 'Imines',
 'Imines',
 'hydrazines',
 'diazo',
 'cyano',
 'containing',
 'methylthio',
 'thiols',
 'thiocarbonyls',
 'halogens',
 't-butyl',
 'trifluoromethyl',
 'acetylenes',
 'cyclopropyl',
 'stuff:',
 'ethoxy',
 'methoxy',
 '???',
 'nitriles']

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem import Draw
from rdkit import Chem
import os
import numpy as np

fName=os.path.join(RDConfig.RDDataDir,'FunctionalGroups.txt')
from rdkit.Chem import FragmentCatalog
fparams = FragmentCatalog.FragCatParams(1,6,fName)

def auto_fg_eval(smiles):
    fcat=FragmentCatalog.FragCatalog(fparams)
    fcgen=FragmentCatalog.FragCatGenerator()
    m = Chem.MolFromSmiles(smiles)
    fcgen.AddFragsFromMol(m,fcat)
    num_entries=fcat.GetNumEntries()
    ans = np.unique(list(fcat.GetEntryFuncGroupIds(num_entries-1))).sort()
    print(ans)
    gt = np.zeros(len(fgroup_names))
    gts = []
    for gggt in gt:
        gts.append('yes' if gggt else 'no')
    return gts



def eval_pope(answers, label_list):
    # label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer#['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )

def main(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    # graph encoder config
    mm_encoder_cfg = MM_ENCODER_CFG(init_checkpoint=args.graph_checkpoint_path)
    mm_encoder_cfg = mm_encoder_cfg.dict()
    # load model
    tokenizer, model, _, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, mm_encoder_cfg=mm_encoder_cfg)
    # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    # self.resize_token_embeddings(len(tokenizer))
    # model.graph_tower = build_graph_tower(LlavaGraphLlamaConfig.from_pretrained(args.model_path))
    model.get_graph_tower()._load_state_dict(args.graph_checkpoint_path,strict=False)
    # model = model.to(dtype=torch.float32)
    # model = model.to(dtype=torch.bfloat16)
    # llava-moleculestm-vicuna-v1-3-7b-pretrain
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
        
    # Input SMILES
    smiles = args.smiles
    while not smiles or not check_smiles_validity(smiles):
        smiles = input("Please enter a valid SMILES: ")
    graph = smiles2graph(smiles,motif=args.motif_augmented)
    graph_tensor = [_convert_dict_to_Data(graph).to(model.device)]

    while True:
        try:
            # inp = "briefly introduce the molecule"#
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if inp.lower() in ["quit", "exit"]:
            print("exit...")
            break
        elif inp == "reset":
            conv = conv_templates[args.conv_mode].copy()
            print("reset conversation...")
            smiles = None
            while not smiles or not check_smiles_validity(smiles):
                smiles = input("Please enter a valid SMILES: ")
            graph = smiles2graph(smiles,motif=args.motif_augmented)
            graph_tensor = [_convert_dict_to_Data(graph).to(model.device)]
           
            continue

        print(f"{roles[1]}: ", end="")

        if graph is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            graph = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # prompt = prompt.replace("\n","")
        # print(prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # print(input_ids)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                graphs=graph_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=128,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/Graph-LLaVA/llava-moleculestm-vicuna-v1-3-7b-pretrain")
    # parser.add_argument("--model-path", type=str, default="./checkpoints/Graph-LLaVA-moleSTMep/llava-moleculestm-vicuna-v1-3-7b-pretrain")
    # parser.add_argument("--graph-checkpoint-path", type=str, default="checkpoints/MoleculeSTM/molecule_model.pth")
    parser.add_argument("--graph-checkpoint-path", type=str, default="checkpoints/MoleculeSTM/demo/demo_checkpoints_Graph/molecule_model.pth")
    parser.add_argument("--model-base", type=str, default="./vicuna-v1-3-7b")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--smiles", type=str, help="SMILES string", default="C([C@H]([C@H]([C@@H]([C@H](CO)O)O)O)O)O")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument('-T',"--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--motif_augmented", action="store_true")
    args = parser.parse_args()
    if "hvqvae" in args.model_path:
        args.motif_augmented = True
    if args.conv_mode is not None:
        args.save_path += args.conv_mode
    if "-vqvae2-" in args.model_path:
        args.graph_checkpoint_path = "./checkpoints/vqencoder_zinc_standard_agent_epoch_60.pth"
    main(args)


"""
    --model-path checkpoints/Graph-LLaVA-4C-hvqvae2-5ep-hlinear-fgprompt-neg-extend-vic/llava-hvqvae2-vicuna-v1-3-7b-pretrain/ \
python -m llava.serve.cli_graph \
    --model-path checkpoints/Graph-LLaVA-4C-hvqvae2-5ep-hlinear-fgprompt-neg-extend/llava-hvqvae2-vicuna-v1-3-7b-pretrain/ \
    --graph-checkpoint-path ./checkpoints/hvqvae300d.pth \
    --model-base ./vicuna-v1-3-7b
"""
