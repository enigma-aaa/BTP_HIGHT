'''
Code from https://github.com/blender-nlp/MolT5

```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''

import pickle
import argparse
import csv
import json
import os.path as osp
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import selfies as sf

def sf_encode(selfies):
    try:
        smiles = sf.decoder(selfies)
        return smiles
    except Exception:
        return None

def convert_to_canonical_smiles(smiles):
    if smiles is None:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
        return canonical_smiles
    else:
        return None
    
def build_evaluate_tuple(result:dict):
    # pred
    # func = lambda x: x.rsplit(']', 1)[0] + ']' if isinstance(x, str) else x
    func = lambda x: x
    result["pred_smi"] = convert_to_canonical_smiles(func(sf_encode(result["pred_self"])))
    # gt
    result["gt_smi"] = convert_to_canonical_smiles(sf_encode(result["gt_self"]))
    return result


def evaluate(input_file, verbose=False):
    outputs = []

    with open(osp.join(input_file)) as f:
        results = json.load(f)
        for i, result in enumerate(results):
            result = build_evaluate_tuple(result)
            gt_self = result['gt_self']
            ot_self = result['pred_self']
            gt_smi = result['gt_smi']
            ot_smi = result['pred_smi']
            if ot_smi is None:
                continue
            outputs.append((result['prompt'], gt_self, ot_self, gt_smi, ot_smi))


    bleu_self_scores = []
    bleu_smi_scores = []

    references_self = []
    hypotheses_self = []
    
    references_smi = []
    hypotheses_smi = []

    for i, (des, gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):

        if i % 100 == 0:
            if verbose:
                print(i, 'processed.')

        gt_self_tokens = [c for c in gt_self]
        out_self_tokens = [c for c in ot_self]

        references_self.append([gt_self_tokens])
        hypotheses_self.append(out_self_tokens)
        
        if ot_smi is None:
            continue
        
        gt_smi_tokens = [c for c in gt_smi]
        ot_smi_tokens = [c for c in ot_smi]

        references_smi.append([gt_smi_tokens])
        hypotheses_smi.append(ot_smi_tokens)
        

    # BLEU score
    bleu_score_self = corpus_bleu(references_self, hypotheses_self)
    if verbose: print(f'SELFIES BLEU score', bleu_score_self)

    references_self = []
    hypotheses_self = []
    
    references_smi = []
    hypotheses_smi = []

    levs_self = []
    levs_smi = []

    num_exact = 0

    bad_mols = 0

    for i, (des, gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):

        hypotheses_self.append(ot_self)
        references_self.append(gt_self)

        hypotheses_smi.append(ot_smi)
        references_smi.append(gt_smi)
        
        try:
            m_out = Chem.MolFromSmiles(ot_smi)
            m_gt = Chem.MolFromSmiles(gt_smi)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1
            #if gt == out: num_exact += 1 #old version that didn't standardize strings
        except:
            bad_mols += 1

        levs_self.append(lev(ot_self, gt_self))
        levs_smi.append(lev(ot_smi, gt_smi))


    # Exact matching score
    exact_match_score = num_exact/(i+1)
    if verbose:
        print('Exact Match:')
        print(exact_match_score)

    # Levenshtein score
    levenshtein_score_smi = np.mean(levs_smi)
    if verbose:
        print('SMILES Levenshtein:')
        print(levenshtein_score_smi)
        
    validity_score = 1 - bad_mols/len(outputs)
    if verbose:
        print('validity:', validity_score)
    
    return exact_match_score, bleu_score_self, levenshtein_score_smi, validity_score 
        
## TEST ##
def test_out_selfies_validity(args):
    with open(osp.join(args.input_file)) as f:
        results = json.load(f)
        bad_selfies = 0
        for i, result in enumerate(results):
            pred = result['pred_self']
            if not sf_encode(pred):
                print(i, pred, 'bad selfies')
                bad_selfies += 1
        print('bad selfies:', bad_selfies)

import argparse
import csv
import os.path as osp
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import selfies as sf


def sf_encode(selfies):
    try:
        smiles = sf.decoder(selfies)
        return smiles
    except Exception:
        return None

def convert_to_canonical_smiles(smiles):
    if smiles is None:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
        return canonical_smiles
    else:
        return None
    
def build_evaluate_tuple(result:dict):
    # pred
    # func = lambda x: x.rsplit(']', 1)[0] + ']' if isinstance(x, str) else x
    func = lambda x: x
    result["pred_smi"] = convert_to_canonical_smiles(func(sf_encode(result["pred_self"])))
    # gt
    result["gt_smi"] = convert_to_canonical_smiles(sf_encode(result["gt_self"]))
    return result
    

def evaluate2(input_file, morgan_r, verbose=False):
    outputs = []
    bad_mols = 0

    with open(osp.join(input_file)) as f:
        results = json.load(f)
        for i, result in enumerate(results):
            result = build_evaluate_tuple(result)
            try:
                gt_smi = result['gt_smi']
                ot_smi = result['pred_smi']
                
                gt_m = Chem.MolFromSmiles(gt_smi)
                ot_m = Chem.MolFromSmiles(ot_smi)

                if ot_m == None: raise ValueError('Bad SMILES')
                outputs.append((result['prompt'], gt_m, ot_m))
            except:
                bad_mols += 1
    validity_score = len(outputs)/(len(outputs)+bad_mols)
    if verbose:
        print('validity:', validity_score)


    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    enum_list = outputs

    for i, (desc, gt_m, ot_m) in enumerate(enum_list):

        if i % 100 == 0:
            if verbose: print(i, 'processed.')

        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    if verbose:
        print('Average RDK Similarity:', rdk_sims_score)
        print('Average MACCS Similarity:', maccs_sims_score)
        print('Average Morgan Similarity:', morgan_sims_score)
        # print(f"{maccs_sims_score}, {rdk_sims_score}, {morgan_sims_score}, {validity_score}")
    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score


## TEST ##
def test_out_selfies_validity(args):
    with open(osp.join(args.input_file)) as f:
        results = json.load(f)
        bad_selfies = 0
        bad_mols = 0
        bad_gt_selfies = 0
        for i, result in enumerate(results):
            pred = result['pred_self']
            smi = sf_encode(pred)
            if not smi:
                bad_selfies += 1
            else:
                try:
                    m = Chem.MolFromSmiles(smi)
                    if m is None:
                        bad_mols += 1
                except:
                    bad_mols += 1
            gt = result['gt_self']
            gt_smi = sf_encode(gt)
            if not gt_smi:
                bad_gt_selfies += 1
        print('Pred: bad selfies:', bad_selfies)
        print('Pred: bad mols:', bad_mols)
        print('GT: bad selfies:', bad_gt_selfies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='caption2smiles_example.json', help='path where test generations are saved')
    parser.add_argument('--morgan_r', type=int, default=2, help='morgan fingerprint radius')
    args = parser.parse_args()
    # test_out_selfies_validity(args)
    exact_match_score, bleu_score_self, levenshtein_score_smi, validity_score =evaluate(args.input_file, verbose=True)
    validity_score2, maccs_sims_score, rdk_sims_score, morgan_sims_score = evaluate2(args.input_file, args.morgan_r, True)
    print(f"{exact_match_score}, {bleu_score_self}, {levenshtein_score_smi}, {validity_score}, {maccs_sims_score}, {rdk_sims_score}, {morgan_sims_score}, {validity_score2}")

"""
python -m llava.eval.molecule_metrics.eval_reaction \
    --input_file
"""
