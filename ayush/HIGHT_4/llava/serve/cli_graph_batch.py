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
import tqdm
from transformers import TextStreamer
from torch_geometric.data import Data
from llava.eval.molecule_metrics.MoleculeNet_classification import _convert_dict_to_Data
from typing import Dict, Optional, Sequence, List
import transformers

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
# def _convert_dict_to_Data(data_dict: Dict) -> Data:
#     return Data(
#         x=torch.asarray(data_dict['node_feat']),
#         edge_attr=torch.asarray(data_dict['edge_feat']),
#         edge_index=torch.asarray(data_dict['edge_index']),
#     )

fgroup_names = ['methyl amide','carboxylic acids','carbonyl methyl ester','terminal aldehyde','amide','carbonyl methyl','isocyanate','isothiocyanate',
 'nitro','nitroso','oximes','Imines','Imines','terminal azo','hydrazines','diazo','cyano',
 'primary sulfonamide','methyl sulfonamide','sulfonic acid','methyl ester sulfonyl',
 'methyl sulfonyl','sulfonyl chloride','methyl sulfinyl','methylthio','thiols','thiocarbonyls',
 'halogens','t-butyl','trifluoromethyl','acetylenes','cyclopropyl',
 'ethoxy','methoxy','side-chain hydroxyls','side-chain aldehydes or ketones','primary amines',
#  '???',
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

def get_fg_answers(smiles):
    fcat=FragmentCatalog.FragCatalog(fparams)
    fcgen=FragmentCatalog.FragCatGenerator()
    m = Chem.MolFromSmiles(smiles)
    fcgen.AddFragsFromMol(m,fcat)
    num_entries=fcat.GetNumEntries()
    gt = np.zeros(len(fgroup_names))
    # print(smiles)
    # print(list(fcat.GetEntryFuncGroupIds(num_entries-1)))
    if num_entries:
        ans = sorted(np.unique(list(fcat.GetEntryFuncGroupIds(num_entries-1))))
        gt[ans] = 1
    gts = []
    for (i,gggt) in enumerate(gt):
        if fgroup_names[i]!="???":
            gts.append('yes' if gggt else 'no')
    # print(ans)
    return gts

def  get_fg_answers_sel(smiles):
    fcat=FragmentCatalog.FragCatalog(fparams)
    fcgen=FragmentCatalog.FragCatGenerator()
    m = Chem.MolFromSmiles(smiles)
    fcgen.AddFragsFromMol(m,fcat)
    num_entries=fcat.GetNumEntries()
    gt = np.zeros(len(fgroup_names))
    gt_neg = np.ones(len(fgroup_names))
    # print(smiles)
    # print(list(fcat.GetEntryFuncGroupIds(num_entries-1)))
    gt_outputs = []
    gt_name_outputs = []
    import random
    if num_entries:
        ans = sorted(np.unique(list(fcat.GetEntryFuncGroupIds(num_entries-1))))
        gt[ans] = 1
        for aa in ans:
            if fgroup_names[aa]!="???":
                gt_outputs.append("yes")
                gt_name_outputs.append(fgroup_names[aa])
        gt_neg[ans] = 0
    
    
    random.seed(len(smiles))
    cnt=6
    while cnt:
        cur = random.choices(np.arange(len(fgroup_names)),weights=gt_neg,k=6)
        for cc in cur:
            if gt[cc]:
                continue
            elif fgroup_names[cc]!="???":
                gt_outputs.append("no")
                gt_name_outputs.append(fgroup_names[cc])
                cnt -= 1
    # gts = []
    # for (i,gggt) in enumerate(gt):
    #     if fgroup_names[i]!="???":
    #         gts.append('yes' if gggt else 'no')
    # print(ans)
    return gt_outputs, gt_name_outputs



def eval_pope(answers, label_list):
    # label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for (i,answer) in enumerate(answers):
        text = answer#['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answers[i] = 'no'
        else:
            answers[i] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer == 'no':
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

    precision = float(TP) / float(TP + FP) if (TP + FP)>0 else -1
    recall = float(TP) / float(TP + FN) if (TP + FN)>0 else -1
    f1 = 2*precision*recall / (precision + recall) if (precision + recall)>0 else -1
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN)>0 else -1
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )
    return (TP, FP, TN, FN), (f1, acc, precision, recall, yes_ratio)

import selfies
def smiles2selfies(smiles_str):
    try:
        selfies_str = selfies.encoder(smiles_str)
    except:
        selfies_str = None
    return selfies_str
    

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
        # tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            print("\n add unk_token \n\n")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(
                    unk_token="<unk>",
                    pad_token="<pad>"
                ),
                tokenizer=tokenizer,
                model=model,
            )
        tokenizer.pad_token = tokenizer.unk_token
    elif 'llama-3' in model_name.lower():
        conv_mode = "llava_llama_3"
        if tokenizer.pad_token is None:
            print("\n add unk_token \n\n")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(
                    unk_token="<unk>",
                    pad_token="<pad>"
                ),
                tokenizer=tokenizer,
                model=model,
            )
        tokenizer.pad_token = tokenizer.unk_token
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    # conv_mode = "llava_llama_2"
    # conv_mode = "plain"
    # tokenizer.pad_token = tokenizer.eos_token
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the batch inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
        
    # Input SMILES
    smiles = args.smiles
    # while not smiles or not check_smiles_validity(smiles):
    #     smiles = input("Please enter a valid SMILES: ")
    graph = smiles2graph(smiles,motif=args.motif_augmented)
    graph_tensor = [_convert_dict_to_Data(graph).to(model.device)]
    smiles_list = [
        'C[C@]12CCC(=O)C=C1CC[C@@H]3[C@@H]2C(=O)C[C@]\\4([C@H]3CC/C4=C/C(=O)OC)C',
        'C[C@H]1[C@H]([C@H]([C@@H]([C@@H](O1)OC[C@@H]2[C@H]([C@@H]([C@H]([C@@H](O2)O)NC(=O)C)O[C@H]3[C@H]([C@@H]([C@@H]([C@@H](O3)C)O)O)O)O[C@H]4[C@@H]([C@H]([C@@H]([C@H](O4)CO)O)O)NC(=O)C)O)O)O',
        'COC1=CC=CC2=C1C(=CN2)C/C(=N/OS(=O)(=O)[O-])/S[C@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO)O)O)O',
        'CN(C)C(=O)C(CCN1CCC(CC1)(C2=CC=C(C=C2)Cl)O)(C3=CC=CC=C3)C4=CC=CC=C4',
        'CC1=C(SC(=[N+]1CC2=CN=C(N=C2N)C)C(CCC(=O)O)O)CCOP(=O)(O)OP(=O)(O)O',
        'C([C@H]([C@H]([C@@H]([C@H](CO)O)O)O)O)O',
        'CCCCCCCCCCCCCCCC/C=C\\OC[C@H](COP(=O)(O)O)O',
        'C1=CC=C(C=C1)[As](=O)(O)[O-]',
        'CCCCCCCCCCCC(=O)OC(=O)CCCCCCCCCCC',
        'CC1=C2C=C(C=C(C2=CC=C1)C(=O)O)[O-]'
    ]
    with open('./data/data/ChEBI-20_data/test.txt','r') as f:
        smiles_list = []
        if args.full:
            lines = f.readlines()[1:]
        else:
            lines = f.readlines()[1:101]
        for line in lines:
            smiles_list.append(line.split('\t')[1])
        print(f"Extracted {len(smiles_list)} from ChEBI-20_data/test.txt.")
    gts = []
    answers = []
    results = []
    cnt = 0
    for smiles in tqdm.tqdm(smiles_list):
        if args.random:
            cur_gts, cur_names = get_fg_answers_sel(smiles)
        else:
            cur_names = fgroup_names
            cur_gts = get_fg_answers(smiles)
        gts += cur_gts
        graph = smiles2graph(smiles,motif=args.motif_augmented)
        graph_tensor = [_convert_dict_to_Data(graph).to(model.device)]
        for (idx,fg_name) in enumerate(cur_names):
            if fg_name=="???":
                continue
            if args.only_pos and labels[cnt+idx]=="no":
                continue
            if args.only_neg and labels[cnt+idx]=="yes":
                continue
            # try:
            # inp = "briefly introduce the molecule"#
            # inp = input(f"{roles[0]}: ")
            inp = f"Is there a {fg_name} group in the molecule?"
            if args.add_selfies:
                selfies_str = smiles2selfies(smiles)
                if selfies_str is not None:
                    inp += f" The compound SELFIES sequence is: {selfies_str}."
            # print(len(labels),cnt+idx)
            # print(f"{roles[0]}: {inp}")
            # print(f"GT:{labels[cnt+idx]}")
            # print(f"{roles[1]}: ", end="")
            losses = []
            for ans in ["Yes","No"]:
                sources=[[{"from":"human","value":f"<image>\n{inp}"},{"from":"gpt","value":f"{ans}"}]]
                from llava.datasets.preprocess import preprocess, preprocess_multimodal
                from llava.constants import IGNORE_INDEX
                import copy
                # sources = preprocess_multimodal(
                #     copy.deepcopy([e["conversations"] for e in sources]),
                #     self.data_args)
                data_dict = preprocess(
                sources,
                tokenizer,
                has_image=True)
                input_ids, labels = data_dict["input_ids"], data_dict["labels"]
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id).to(model.device)
                labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                        batch_first=True,
                                                        padding_value=IGNORE_INDEX).to(model.device)
                input_ids = input_ids[:, :tokenizer.model_max_length]
                labels = labels[:, :tokenizer.model_max_length]
                batch = dict(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=input_ids.ne(tokenizer.pad_token_id),
                )
                batch['graphs'] = graph_tensor
                with torch.inference_mode():
                    loss = model(input_ids=input_ids,labels=labels,attention_mask=input_ids.ne(tokenizer.pad_token_id),graphs=graph_tensor).loss.item()
                losses.append(loss)
            outputs = 'yes' if losses[0]<losses[1] else 'no'
            result={"smiles":smiles,"prompt": inp, "outputs": outputs, "gt": cur_gts[idx],'losses':losses}
            if args.debug:
                print("\n", result, "\n")
            answers.append(outputs)
            results.append(result)
        cnt += len(fgroup_names)
    import json
    save_file_path = f"cli_batch_"
    if args.random:
        save_file_path = f"cli_batch_random_"
    if args.full:
        save_file_path = f"cli_batch_full_"
    if args.only_pos:
        save_file_path = f"cli_batch_pos_"
        gts = ['yes']*len(answers)
    if args.only_neg:
        save_file_path = f"cli_batch_neg_"
        gts = ['no']*len(answers)
    save_file_path += f"{args.save_path }_smiles{len(smiles_list)}"
    with open(f"{save_file_path}.jsonl","w") as ans_file:
        json.dump(results, ans_file, indent=2)
    eval_results = eval_pope(answers,copy.deepcopy(gts))
    with open(f"{save_file_path}.log","w") as ans_file:
        ans_file.write('TP\tFP\tTN\tFN\t\n')
        ans_file.write('{}\t{}\t{}\t{}\n'.format(eval_results[0][0], eval_results[0][1], eval_results[0][2], eval_results[0][3]))
        ans_file.write('%.3f, %.3f, %.3f, %.3f, %.3f\n' % (eval_results[1][0], eval_results[1][1], eval_results[1][2], eval_results[1][3], eval_results[1][4]))
    
    answers_pos = []
    answers_neg = []
    # print(gts)
    for (i,ll) in enumerate(gts):
        if ll.lower() == "yes":
            answers_pos.append(answers[i])
        else:
            if answers[i].lower() == "no":
                answers_neg.append("yes")
            else:
                answers_neg.append("no")
    print("positive class")
    eval_results = eval_pope(answers_pos,["yes"]*len(answers_pos))
    with open(f"{save_file_path}.log","a") as ans_file:
        ans_file.write('positive class\n')
        ans_file.write('TP\tFP\tTN\tFN\t\n')
        ans_file.write('{}\t{}\t{}\t{}\n'.format(eval_results[0][0], eval_results[0][1], eval_results[0][2], eval_results[0][3]))
        ans_file.write('%.3f, %.3f, %.3f, %.3f, %.3f\n' % (eval_results[1][0], eval_results[1][1], eval_results[1][2], eval_results[1][3], eval_results[1][4]))
    
    print("negative class")
    eval_results = eval_pope(answers_neg,["yes"]*len(answers_neg))
    with open(f"{save_file_path}.log","a") as ans_file:
        ans_file.write('negative class\n')
        ans_file.write('TP\tFP\tTN\tFN\t\n')
        ans_file.write('{}\t{}\t{}\t{}\n'.format(eval_results[0][0], eval_results[0][1], eval_results[0][2], eval_results[0][3]))
        ans_file.write('%.3f, %.3f, %.3f, %.3f, %.3f\n' % (eval_results[1][0], eval_results[1][1], eval_results[1][2], eval_results[1][3], eval_results[1][4]))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/Graph-LLaVA-4C-hvqvae2-3ep/llava-hvqvae2-vicuna-v1-3-7b-pretrain")
    parser.add_argument("--save-path", type=str, default="")
    # parser.add_argument("--model-path", type=str, default="./checkpoints/Graph-LLaVA-moleSTMep/llava-moleculestm-vicuna-v1-3-7b-pretrain")
    # parser.add_argument("--graph-checkpoint-path", type=str, default="checkpoints/MoleculeSTM/molecule_model.pth")
    parser.add_argument("--graph-checkpoint-path", type=str, default="./checkpoints/hvqvae300d.pth")
    parser.add_argument("--model-base", type=str, default="./vicuna-v1-3-7b")
    # parser.add_argument("--conv-mode", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--smiles", type=str, help="SMILES string", default="C([C@H]([C@H]([C@@H]([C@H](CO)O)O)O)O)O")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument('-T',"--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--motif_augmented", action="store_true")
    parser.add_argument("--add_selfies", action="store_true")
    parser.add_argument("--only_pos", action="store_true")
    parser.add_argument("--only_neg", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if "hvqvae" in args.model_path:
        args.motif_augmented = True
    if "add_seqs" in args.model_path:
        args.add_selfies = True
    # print(args.model_path.split('/'))
    if len(args.save_path)==0:
        args.save_path = args.model_path.split('/')[-3] if "Graph-LLaVA" in args.model_path.split('/')[-3]  else args.model_path.split('/')[-2]
    if "llama-2" in args.model_base:
        args.save_path += args.model_base
    if "llama-3" in args.model_base:
        args.save_path += args.model_base
    if args.conv_mode is not None:
        args.save_path += args.conv_mode
    if "-vqvae2-" in args.model_path:
        args.graph_checkpoint_path = "./checkpoints/vqencoder_zinc_standard_agent_epoch_60.pth"
    # print(args.model_path.split('/'))
    print(f"results will be saved to {args.save_path }")
    main(args)


"""
python -m llava.serve.cli_graph \
    --model-path checkpoints/Graph-LLaVA/molcap-llava-moleculestm-vicuna-v1-3-7b-finetune_lora \
    --graph-checkpoint-path checkpoints/MoleculeSTM/molecule_model.pth \
    --model-base checkpoints/vicuna-v1-3-7b \

python -m llava.serve.cli_graph \
    --model-base checkpoints/llama-2-7b-chat \
"""
