import json
import argparse
from sklearn.metrics import mean_absolute_error
from typing import List

LUMOs=[
    'Please provide the lowest unoccupied molecular orbital (LUMO) energy of this molecule.',
    'Please provide me with the LUMO energy value of this molecule.',
    'What is the LUMO energy of this molecule?',
    'I would like to know the LUMO energy of this molecule, could you please provide it?',
    'I would like to know the lowest unoccupied molecular orbital (LUMO) energy of this molecule, could you please provide it?',
    'Please provide the LUMO energy value for this molecule.',
    'I am interested in the LUMO energy of this molecule, could you tell me what it is?',
    'Could you give me the LUMO energy value of this molecule?',
    'Can you tell me the value of the LUMO energy for this molecule?',
    'What is the LUMO level of energy for this molecule?',
    'What is the lowest unoccupied molecular orbital (LUMO) energy of this molecule?',
    'Please provide the lowest unoccupied molecular orbital (LUMO) energy value for this molecule.',
]

HOMOs=[
    'I would like to know the highest occupied molecular orbital (HOMO) energy of this molecule, could you please provide it?',
    'I am interested in the HOMO energy of this molecule, could you tell me what it is?',
    'Can you tell me the value of the HOMO energy for this molecule?',
    'I would like to know the HOMO energy of this molecule, could you please provide it?',
    'Please provide me with the HOMO energy value of this molecule.',
    'Please provide the highest occupied molecular orbital (HOMO) energy of this molecule.',
    'What is the HOMO level of energy for this molecule?',
    'What is the HOMO energy of this molecule?',
    'Could you give me the HOMO energy value of this molecule?',
    'Please provide the HOMO energy value for this molecule.',
    'What is the highest occupied molecular orbital (HOMO) energy of this molecule?',
    'Please provide the highest occupied molecular orbital (HOMO) energy value for this molecule.',
]

HOMO_LUMOs=[
    'Please provide the energy separation between the highest occupied and lowest unoccupied molecular orbitals (HOMO-LUMO gap) of this molecule.',
    'Can you give me the energy difference between the HOMO and LUMO orbitals of this molecule?',
    'What is the energy separation between the HOMO and LUMO of this molecule?',
    'I need to know the HOMO-LUMO gap energy of this molecule, could you please provide it?',
    'What is the HOMO-LUMO gap of this molecule?',
    'Please provide the gap between HOMO and LUMO of this molecule.',
    'I would like to know the HOMO-LUMO gap of this molecule, can you provide it?',
    'Please give me the HOMO-LUMO gap energy for this molecule.',
    'Could you tell me the energy difference between HOMO and LUMO for this molecule?'
]

def is_prompt_type(prompt,templates):
    for temp in templates:
        if temp in prompt:
            return True
    return False

def compute_mae(eval_result_file:str, except_idxs:List[int]=[],mode="all"):
    with open(eval_result_file) as f:
        results = json.load(f)
        gts = []
        preds = []
        preds_except = []
        for i, result in enumerate(results):
            if i in except_idxs:
                continue
            pred = result['pred_self']
            gt = result['gt_self']
            if mode=="all" or (mode=="HOMOs" and is_prompt_type(result["prompt"],HOMOs)) \
                or (mode=="LUMOs" and is_prompt_type(result["prompt"],LUMOs))or (mode=="HOMO_LUMOs" and is_prompt_type(result["prompt"],HOMO_LUMOs)):
                try:
                    predd=float(pred.split("\n")[0].split("\t")[0])
                    gts.append(float(gt))
                    preds.append(predd)
                except Exception as e:
                    preds_except.append(f"{i}:{gt}\t{pred}")
        print(len(gts),len(preds))
        print(f"Exception samples {len(preds_except)}")
        # print(preds_except)
        return mean_absolute_error(gts, preds)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_result_file", type=str, required=True)
    args = parser.parse_args()
    # read except_idxs
    # with open('./Mol-Instructions/Molecule-oriented_Instructions/property_overlap.txt', 'r') as f:
    #     except_idxs = [int(line.split('\t')[0]) for line in f.readlines()]
    except_idxs = []
    res = []
    for mode in ["HOMOs","LUMOs","HOMO_LUMOs","all"]:
        mae = compute_mae(args.eval_result_file, except_idxs,mode=mode)
        print(mode,mae)
        res.append(mae)
    print(res)
    
    
"""
# InstructMol/eval_result/hvqvae-origdata-ep3-property_pred-5ep.jsonl
# InstructMol/eval_result/moleculestm-origdata-ep3-property_pred-5ep.jsonl
# property_pred
TASK=property_pred
EPOCH=5
GRAPH_TOWER=vqvae
python -m llava.eval.molecule_metrics.property_metrics \
    --eval_result_file=eval_result/$GRAPH_TOWER-$TASK-${EPOCH}ep.jsonl | tee 0318${GRAPH_TOWER}_property_pred.log
"""
