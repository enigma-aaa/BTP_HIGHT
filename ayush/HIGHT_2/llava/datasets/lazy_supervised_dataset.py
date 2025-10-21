import os
import json
import copy
import pickle
from PIL import Image
from typing import Dict, Optional, Sequence, List

import torch
from torch.utils.data import Dataset
import transformers
from .preprocess import preprocess, preprocess_multimodal
import random
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict
    
import pandas as pd
from llava.datasets.smiles2graph import construct_instruct_question, smiles2graph
import tqdm
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem import Draw
from rdkit import Chem
import os
import numpy as np
class LazySupervisedGraphDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedGraphDataset, self).__init__()
        self.data_args = data_args
        
        if "ChEBI".lower() in data_path.lower():
            print(f"Loading {data_path}")
            data = open(data_path,"r").readlines()
            self.list_data_dict = []
            for line in tqdm.tqdm(data[1:],desc="converting"):
                line=line.split("\t")
                # print(line)
                smiles = line[1]
                desc = " ".join(line[2:])
                cur_data = {}
                cur_data["conversations"] = [{"from":"human","value":construct_instruct_question()},{"from":"gpt","value":desc}]
                import selfies
                if data_args.add_selfies:
                    try:
                        selfies_seq = selfies.encoder(smiles,strict=False)
                        cur_data["conversations"][0]["value"] += f" The compound SELFIES sequence is: {selfies_seq}"
                    except Exception as e:
                        print(e)
                # cur_data["graph"]=smiles2graph(selfies.decoder(smiles))
                # print(smiles)
                try:
                    cur_data["graph"] = smiles2graph(smiles,motif=data_args.motif_augmented,add_lap_pe=data_args.add_lap_pe)
                    self.list_data_dict.append(cur_data)
                except Exception as e:
                    continue
            # bi_file = "./Mol-Instructions/data/ChEBI-20_data/data_dict_train.pkl"
            # pickle.dump(self.list_data_dict,open(bi_file,"wb"))
        else:
            self.root = "./data/PubChemSTM_data"
            bi_file = "data_dict"
            if data_args.motif_augmented:
                bi_file = "hi_"+bi_file
            if data_args.add_selfies:
                bi_file += "_self"
            if data_args.add_lap_pe:
                bi_file += "_lap"
            if self.data_args.add_fg_prompt:
                bi_file += "_fgprompt"
            bi_file = os.path.join(self.root,f"{bi_file}.pkl")
            if not os.path.exists(bi_file):
                CID2text_file = os.path.join(self.root, "raw/CID2text.json")
                CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")
                self.load_CID2SMILES(CID2text_file, CID2SMILES_file)
                self.list_data_dict = []
                # self.list_data_dict_old = pickle.load(open(bi_file.replace("_self",""),"rb"))
                i=-1
                for CID in tqdm.tqdm(self.CID2SMILES.keys(),desc="preprocessing"):
                    i+=1
                    # if i<48272:
                    #     continue
                    
                    cur_data = {}
                    conversations = self.CID2text_data[CID]
                    cur_data["conversations"] = [{"from":"human","value":construct_instruct_question()},{"from":"gpt","value":conversations[0]}]
                    cur_data["smiles"] = self.CID2SMILES[CID]
                    if data_args.motif_augmented:
                        import selfies
                        try:
                            selfies_seq = selfies.encoder(self.CID2SMILES[CID],strict=False)
                            if data_args.add_selfies:
                                cur_data["conversations"][0]["value"] += f" The compound SELFIES sequence is: {selfies_seq}"
                            cur_data["graph"] = smiles2graph(self.CID2SMILES[CID],motif=data_args.motif_augmented,add_lap_pe=data_args.add_lap_pe)
                            if self.data_args.add_fg_prompt:
                                fg_prompt = self.get_fg_prompt(self.CID2SMILES[CID],num_part=cur_data["graph"].get("num_part",None))
                                cur_data['conversations'][1]['value']= f"{fg_prompt} {cur_data['conversations'][1]['value']}"
                            # cur_data["graph"]=self.list_data_dict_old[i]["graph"]
                            self.list_data_dict.append(cur_data)
                        except Exception as e:
                            print(e)
                            # continue
                    # if i>=1000:
                    #     break
                    
                pickle.dump(self.list_data_dict,open(bi_file,"wb"))
            else:
                print(f"load stage 1 data from {bi_file}")
                with open(bi_file,"rb") as f:
                    self.list_data_dict = pickle.load(f)


        self.tokenizer = tokenizer
        self.data_args = data_args
    def load_CID2SMILES(self, CID2text_file, CID2SMILES_file):
        with open(CID2text_file, "r") as f:
            self.CID2text_data = json.load(f)
        print("len of CID2text: {}".format(len(self.CID2text_data.keys())))

        df = pd.read_csv(CID2SMILES_file)
        CID_list, SMILES_list = df["CID"].tolist(), df["SMILES"].tolist()
        self.CID2SMILES = {}
        for CID, SMILES in zip(CID_list, SMILES_list):
            CID = str(CID)
            self.CID2SMILES[CID] = SMILES
        print("len of CID2SMILES: {}".format(len(self.CID2SMILES.keys())))
        return
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.data_args.add_fg_prompt_neg2:
            sources = self.list_data_dict[i]
        else:
            sources = copy.deepcopy(self.list_data_dict[i])
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        import random
        if self.data_args.add_fg_prompt_neg2:
            # if random.random() <= 0.5:
            if " atoms and " in sources[0]['conversations'][1]['value'] or " 0 functional groups." in sources[0]['conversations'][1]['value']:
                sources[0]['conversations'][1]['value'] = ".".join(sources[0]['conversations'][1]['value'].split(".")[1:])
            sources[0]['conversations'][1]['value'] = self.get_fg_prompt(sources[0]['smiles'])+" "+self.get_fg_prompt_neg(sources[0]['smiles'])+\
                                                        sources[0]['conversations'][1]['value']
            # question, answer = self.get_fg_prompt_bin(sources[0]['smiles'])
            # sources[0]['conversations'][1]['value'] = answer
            # sources[0]['conversations'][0]['value'] = "<image>\n" +question
        if self.data_args.add_fg_prompt:
            if random.random() <= 0.5:
                if " atoms and " in sources[0]['conversations'][1]['value'] or " 0 functional groups." in sources[0]['conversations'][1]['value']:
                    sources[0]['conversations'][1]['value'] = ".".join(sources[0]['conversations'][1]['value'].split(".")[1:])
                # sources[0]['conversations'][1]['value'] = self.get_fg_prompt(sources[0]['smiles'])+" "+self.get_fg_prompt_neg(sources[0]['smiles'])+\
                #                                             sources[0]['conversations'][1]['value']
                question, answer = self.get_fg_prompt_bin(sources[0]['smiles'])
                sources[0]['conversations'][1]['value'] = answer
                sources[0]['conversations'][0]['value'] = "<image>\n" +question
            # for ss in sources:
            #     print(ss['conversations'])
            #     # exit()
        if self.data_args.extend_fg_prompt:
            import random
            if random.random() <= 0.33:
                question_pools = [
                    'Could you give me a brief overview of the functional groups in this molecule?',
                    'Could you provide a description of the functional groups in this molecule?',
                    'Describe the functional groups in this molecule.',
                    'Please give me some details about the functional groups in this molecule.',
                    'Provide a brief overview of the functional groups in this molecule.',
                    'Provide a description of the functional groups in this molecule.',
                    'What can you tell me about the functional groups in this molecule?'
                ]
                question = random.choice(question_pools)
                # if " atoms and " in sources[0]['conversations'][1]['value'] or " 0 functional groups." in sources[0]['conversations'][1]['value']:
                #     sources[0]['conversations'][1]['value'] = ".".join(sources[0]['conversations'][1]['value'].split(".")[1:])
                sources[0]['conversations'][1]['value'] = self.get_fg_prompt(sources[0]['smiles'])+" "+self.get_fg_prompt_neg(sources[0]['smiles'])
                sources[0]['conversations'][0]['value'] = "<image>\n" +question
            elif random.random() <= 0.66:
                question_pools = [
                    'Could you give me some functional groups not present in this molecule?',
                    'Describe some functional groups that do not exist in this molecule.',
                    'Please give me some functional groups not present in this molecule.',
                    'Provide some functional groups not present in this molecule.',
                    'Provide a description of the functional groups not present in this molecule.',
                    'What can you tell me about the functional groups that do not exist in this molecule?'
                ]
                question = random.choice(question_pools)
                # if " atoms and " in sources[0]['conversations'][1]['value'] or " 0 functional groups." in sources[0]['conversations'][1]['value']:
                #     sources[0]['conversations'][1]['value'] = ".".join(sources[0]['conversations'][1]['value'].split(".")[1:])
                sources[0]['conversations'][1]['value'] = self.get_fg_prompt_neg(sources[0]['smiles'],all=True)
                sources[0]['conversations'][0]['value'] = "<image>\n" +question
            else:
                # use the original
                pass
        # print(sources)
        # exit()
        if 'graph' in sources[0]:
            graph = self.list_data_dict[i]['graph']
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('graph' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # graph exist in the data
        if 'graph' in self.list_data_dict[i]:
            data_dict['graph'] = graph
        elif self.data_args.is_multimodal:
            raise ValueError("Graph does not exist in the data, but the model is multimodal")
        return data_dict

    def get_fg_prompt(self,smiles,num_part=None):
        fgroup_names = ['methyl amide',
                        'carboxylic acids',
                        'carbonyl methyl ester',
                        'terminal aldehyde',
                        'amide','carbonyl methyl','isocyanate','isothiocyanate',
                        'nitro','nitroso','oximes','Imines','Imines','terminal azo','hydrazines','diazo','cyano',
                        'primary sulfonamide','methyl sulfonamide','sulfonic acid','methyl ester sulfonyl',
                        'methyl sulfonyl','sulfonyl chloride','methyl sulfinyl','methylthio','thiols','thiocarbonyls',
                        'halogens','t-butyl','trifluoromethyl','acetylenes','cyclopropyl',
                        'ethoxy','methoxy','side-chain hydroxyls','side-chain aldehydes or ketones','primary amines',
                        '???',
                        'nitriles']
        fgroup_intro = [
            "The methyl amide group is a functional group consisting of a methyl group (-CH3) attached to an amide group (-CONH2). It can be represented as -CONHCH3. This group is characterized by the presence of a carbonyl group (C=O) bonded to a nitrogen atom (N) which, in turn, is bonded to a methyl group. The methyl amide group is commonly found in organic molecules and can influence the chemical properties and reactivity of compounds due to its polar nature and ability to participate in hydrogen bonding.",
            "Carboxylic acids are organic compounds characterized by the presence of a carboxyl group (-COOH), which consists of a carbonyl group (C=O) attached to a hydroxyl group (-OH). These acids are typically weak acids, capable of donating a proton (H⁺) due to the polar nature of the carboxyl group. The general formula for carboxylic acids is R-COOH, where 'R' represents a hydrocarbon chain or hydrogen. Carboxylic acids are commonly found in nature and are key intermediates in various biochemical pathways, including the citric acid cycle. They exhibit unique chemical properties such as the ability to form hydrogen bonds, making them soluble in water and capable of forming dimers in the gas phase.",
            "The carbonyl methyl ester group is a functional group in organic chemistry characterized by the presence of a carbonyl group (C=O) bonded to an oxygen atom which, in turn, is bonded to a methyl group (-OCH3). This group is derived from carboxylic acids where the hydrogen of the hydroxyl group is replaced by a methyl group. The general formula for a carbonyl methyl ester is R-COOCH3, where 'R' represents a hydrocarbon chain or hydrogen. Carbonyl methyl esters are known for their pleasant fragrances and are commonly used in the production of flavors and fragrances. They are also important intermediates in organic synthesis and can undergo hydrolysis to revert to the original carboxylic acid and methanol.",
            "The terminal aldehyde group is a functional group in organic chemistry characterized by the presence of a carbonyl group (C=O) bonded to at least one hydrogen atom and located at the end of a carbon chain. Its general formula is R-CHO, where 'R' represents a hydrocarbon chain or hydrogen. In this group, the carbonyl carbon is bonded to a hydrogen atom, making it distinct from ketones, where the carbonyl carbon is bonded to two carbon atoms. Aldehydes are typically more reactive than ketones due to the presence of the hydrogen atom, which makes the carbonyl carbon more electrophilic. They play crucial roles in various biochemical processes, including as intermediates in metabolic pathways and in the formation of more complex molecules. Aldehydes can be oxidized to carboxylic acids and reduced to primary alcohols, highlighting their versatility in chemical reactions.",
            "An amide group is a functional group characterized by a carbonyl group (C=O) linked to a nitrogen atom (N). It has the general structure R-CO-NR'R'', where R, R', and R'' can be hydrogen atoms or organic groups. Amides are formed through the condensation reaction of a carboxylic acid and an amine. The carbonyl carbon is bonded to the nitrogen, which distinguishes it from other carbonyl-containing groups like esters and ketones. Amides are commonly found in proteins where they link amino acids via peptide bonds. They exhibit resonance, which imparts partial double-bond character to the C-N bond, making it less reactive than typical single bonds.",
            "A carbonyl methyl group refers to a methyl group (CH3) attached to a carbonyl group (C=O). This functional group is commonly found in compounds like acetone (CH3-CO-CH3), where the carbonyl carbon is bonded to two methyl groups. The presence of the carbonyl group significantly affects the chemical reactivity and properties of the molecule, often making it more electrophilic and susceptible to nucleophilic attacks.",
            "The isocyanate group is a functional group in organic chemistry characterized by the structure -N=C=O, where a nitrogen atom is doubly bonded to a carbon atom, which is also doubly bonded to an oxygen atom. Isocyanates are highly reactive compounds used primarily in the production of polyurethanes, where they react with alcohols (polyols) to form urethane linkages. They can also react with amines to form ureas and with water to form amines and carbon dioxide. Due to their reactivity, isocyanates are potent electrophiles and can pose health hazards, including respiratory issues and sensitization, necessitating careful handling and use.",
            "The isothiocyanate group is a functional group in organic chemistry characterized by the structure -N=C=S, where a nitrogen atom is doubly bonded to a carbon atom, which is doubly bonded to a sulfur atom. Isothiocyanates are derived from the decomposition of glucosinolates found in cruciferous vegetables and are known for their pungent taste and aroma. They are biologically active compounds with potential health benefits, including anticancer properties, due to their ability to induce phase II detoxification enzymes. Isothiocyanates are also used in organic synthesis and as intermediates in the production of pharmaceuticals and agrochemicals.",
            "The nitro group is a functional group in organic chemistry characterized by the structure -NO2, where a nitrogen atom is bonded to two oxygen atoms, one via a single bond and the other via a double bond. This group is highly electron-withdrawing due to the presence of the electronegative oxygen atoms, which impart significant polarity to the molecule. Nitro groups are commonly found in explosives, pharmaceuticals, and dyes. In biochemistry, nitro compounds can act as intermediates in various metabolic pathways and are sometimes used as antibiotics or drugs. The nitro group can undergo reduction to form amines, making it a versatile functional group in synthetic chemistry.",
            "The nitroso group is a functional group in organic chemistry with the formula -NO, consisting of a nitrogen atom double-bonded to an oxygen atom and single-bonded to a carbon atom of an organic molecule. It is characterized by its ability to participate in various chemical reactions, including electrophilic addition and radical formation. The nitroso group is often found in compounds used for industrial applications, pharmaceuticals, and as intermediates in organic synthesis.",
            "The oximes group is a functional group in organic chemistry with the general formula R1R2C=NOH, where R1 and R2 can be hydrogen, alkyl, or aryl groups. It is characterized by the presence of a carbon-nitrogen double bond (C=N) with an attached hydroxyl group (OH). Oximes are typically formed by the reaction of hydroxylamine (NH2OH) with aldehydes or ketones and are used in various applications, including the identification of carbonyl compounds, as intermediates in organic synthesis, and in pharmaceuticals for their potential therapeutic properties.",
            "The imines group, also known as Schiff bases, is a functional group in organic chemistry with the general formula R1R2C=NR3, where R1 and R2 can be hydrogen, alkyl, or aryl groups, and R3 is typically a hydrogen atom or an organic substituent. It is characterized by a carbon-nitrogen double bond (C=N). Imines are formed by the condensation reaction of primary amines with aldehydes or ketones. They play crucial roles in various biochemical processes, including as intermediates in organic synthesis, in the formation of certain biomolecules, and in medicinal chemistry for drug development.",
            "The imines group, also known as Schiff bases, is a functional group in organic chemistry with the general formula R1R2C=NR3, where R1 and R2 can be hydrogen, alkyl, or aryl groups, and R3 is typically a hydrogen atom or an organic substituent. It is characterized by a carbon-nitrogen double bond (C=N). Imines are formed by the condensation reaction of primary amines with aldehydes or ketones. They play crucial roles in various biochemical processes, including as intermediates in organic synthesis, in the formation of certain biomolecules, and in medicinal chemistry for drug development.",
            "The terminal azo group is a functional group in organic chemistry with the general structure R-N=N-H, where R represents an organic substituent. It features a nitrogen-nitrogen double bond (N=N) with one nitrogen atom bonded to an organic group and the other to a hydrogen atom. Terminal azo compounds are typically less stable than their non-terminal counterparts and can participate in various chemical reactions, including reduction and coupling reactions. They are commonly used in dye chemistry, molecular switches, and sometimes in pharmaceuticals for their unique reactivity and properties.",
            "The hydrazines group is a functional group in organic chemistry with the general formula R2N-NR2, where R can be hydrogen, alkyl, or aryl groups. It consists of two nitrogen atoms single-bonded to each other, with each nitrogen bearing one or two substituents. Hydrazines are characterized by their high reactivity and are used in various applications, including as intermediates in organic synthesis, in pharmaceuticals for their potential therapeutic effects, and as propellants in rocket fuels due to their energetic properties.",
            "The diazo group is a functional group in organic chemistry with the general formula R2C=N2, where R represents an organic substituent. It features a carbon atom double-bonded to two nitrogen atoms, forming a unique carbon-nitrogen-nitrogen (C=N=N) structure. Diazo compounds are known for their high reactivity and are commonly used in organic synthesis for introducing nitrogen groups, in cyclopropanation reactions, and as intermediates in the preparation of azo dyes. Their ability to release nitrogen gas makes them valuable in various chemical transformations.",
            "The cyano group is a functional group in organic chemistry with the formula -C≡N, consisting of a carbon atom triple-bonded to a nitrogen atom. It is highly electronegative and polar, often increasing the reactivity of the molecule it is attached to. The cyano group is found in nitriles, where it is bonded to an alkyl or aryl group, and it plays a crucial role in organic synthesis, pharmaceuticals, and materials science due to its ability to participate in various chemical reactions, including nucleophilic addition and polymerization.",
            "The primary sulfonamide group is a functional group in organic chemistry with the general formula R-SO2-NH2, where R represents an organic substituent. It consists of a sulfonyl group (SO2) bonded to an amino group (NH2). This group is known for its high polarity and ability to form strong hydrogen bonds, which makes it important in medicinal chemistry for designing drugs with good solubility and binding properties. Primary sulfonamides are widely used in pharmaceuticals, particularly as antibiotics and enzyme inhibitors, due to their ability to interact with biological targets.",
            "The methyl sulfonamide group is a functional group in organic chemistry with the formula CH3-SO2-NH2. It consists of a methyl group (CH3) attached to a sulfonyl group (SO2), which is in turn bonded to an amino group (NH2). This group is highly polar and can form strong hydrogen bonds, making it useful in medicinal chemistry for enhancing the solubility and binding affinity of drug molecules. Methyl sulfonamides are often employed as intermediates in organic synthesis and in pharmaceuticals for their ability to modulate biological activity.",
            "The sulfonic acid group, with the general formula R-SO3H, is a functional group in organic chemistry characterized by a sulfur atom bonded to three oxygen atoms (one of which is double-bonded) and a hydroxyl group. It is highly polar and strongly acidic, often used in detergents, dyes, and pharmaceuticals due to its ability to enhance solubility and reactivity in aqueous solutions.",
            "The methyl ester sulfonyl group, with the general formula R-SO2-OMe, consists of a sulfonyl group (R-SO2) bonded to a methoxy group (OMe). This functional group combines the properties of sulfonyl groups, which are highly polar and electron-withdrawing, with the characteristics of esters, making it useful in organic synthesis and medicinal chemistry for modifying the reactivity and solubility of compounds.",
            "The methyl sulfonyl group, with the formula CH3-SO2-, consists of a methyl group (CH3-) bonded to a sulfonyl group (SO2-). This functional group is highly polar and electron-withdrawing, often used in organic synthesis and medicinal chemistry to modify the chemical properties and reactivity of molecules.",
            "The sulfonyl chloride group, with the general formula R-SO2Cl, consists of a sulfonyl group (SO2) bonded to a chlorine atom. This highly reactive functional group is commonly used as an intermediate in the synthesis of sulfonamides, sulfonate esters, and other sulfur-containing compounds in both organic and medicinal chemistry.",
            "The methyl sulfinyl group, with the formula CH3-S(O)-, consists of a methyl group (CH3-) bonded to a sulfinyl group (S=O). This functional group is characterized by a sulfur atom double-bonded to an oxygen atom and bonded to a methyl group, making it polar and useful in various chemical reactions and as an intermediate in organic synthesis.",
            "The methylthio group is a functional group in organic chemistry with the formula -S-CH3. It consists of a sulfur atom bonded to a methyl group (CH3). This group is known for its electron-donating properti   es and is often used to modify the chemical and biological properties of molecules. Methylthio groups are found in various natural products and pharmaceuticals, where they can influence stability, reactivity, and biological activity by altering electronic and steric properties.",
            "The thiols group, also known as mercaptans, is a functional group in organic chemistry with the formula R-SH, where R represents an organic substituent. It consists of a sulfur atom bonded to a hydrogen atom. Thiols are characterized by their strong, often unpleasant odor and their ability to form disulfide bonds (R-S-S-R) through oxidation, which is crucial in the stabilization of protein structures. They are highly reactive, capable of participating in various chemical reactions, and play significant roles in biochemistry, including enzyme function and redox regulation.",
            "The thiocarbonyls group is a functional group in organic chemistry with the general formula R2C=S, where R can be hydrogen, alkyl, or aryl groups. It consists of a carbon atom double-bonded to a sulfur atom. Thiocarbonyls are analogs of carbonyl groups (C=O) but with sulfur replacing oxygen, which imparts different reactivity due to the larger size and lower electronegativity of sulfur. They are found in thioesters, thioketones, and thioureas, playing roles in organic synthesis and biochemical processes, including as intermediates in various chemical reactions and in the regulation of enzyme activities.",
            "The halogens group consists of the elements fluorine, chlorine, bromine, iodine, and astatine, which are found in Group 17 of the periodic table. These elements are highly reactive nonmetals with seven valence electrons, making them eager to gain an electron to achieve a stable octet configuration. In organic chemistry, halogens are often attached to carbon atoms, forming organohalides. Halogenation can significantly alter the chemical properties of molecules, affecting reactivity, polarity, and biological activity. Halogens play crucial roles in various applications, including pharmaceuticals, agrochemicals, and materials science.",
            "The t-butyl group, also known as tert-butyl or t-Bu, is a bulky alkyl substituent with the formula (CH3)3C-. It consists of a central carbon atom bonded to three methyl groups (CH3) and one variable group. This branching structure makes the t-butyl group sterically hindered, which can influence the reactivity and stability of molecules it is attached to by providing steric protection. The t-butyl group is commonly used in organic synthesis to protect functional groups and in medicinal chemistry to enhance the pharmacokinetic properties of drugs by increasing metabolic stability.",
            "The trifluoromethyl group, with the formula -CF3, consists of a carbon atom bonded to three fluorine atoms. This group is highly electronegative and electron-withdrawing due to the presence of fluorine atoms, which significantly affects the chemical and physical properties of the molecules it is attached to. The trifluoromethyl group is commonly used in medicinal chemistry to enhance the metabolic stability, lipophilicity, and bioavailability of pharmaceuticals. It also plays a role in agrochemicals and materials science, where it can improve the performance and durability of compounds.",
            "The acetylenes group, also known as alkynes, is a functional group in organic chemistry with the general formula R1-C≡C-R2, where R1 and R2 can be hydrogen, alkyl, or aryl groups. It features a carbon-carbon triple bond (C≡C), which is linear and highly reactive due to the presence of two π-bonds. Acetylenes are used in organic synthesis for constructing complex molecules, in pharmaceuticals for their potential biological activities, and in materials science for producing polymers and specialty chemicals. Their unique structure imparts significant electronic and steric properties that influence their reactivity and applications.",
            "The cyclopropyl group is a functional group in organic chemistry with the formula -C3H5, consisting of a three-membered carbon ring. This ring is highly strained due to its 60-degree bond angles, making it more reactive than larger cycloalkanes. The cyclopropyl group can influence the chemical and biological properties of molecules it is attached to, often increasing rigidity and affecting metabolic stability. It is used in medicinal chemistry to enhance the pharmacokinetic properties of drugs and in organic synthesis to introduce strain and reactivity into molecules.",
            "The ethoxy group is a functional group in organic chemistry with the formula -OCH2CH3. It consists of an oxygen atom bonded to an ethyl group (CH2CH3). The ethoxy group is an ether functional group, contributing to the molecule's overall polarity and affecting its solubility and reactivity. It is often used in organic synthesis to modify the physical and chemical properties of compounds, and in medicinal chemistry to improve the pharmacokinetic properties of drugs, such as solubility and metabolic stability.",
            "The methoxy group is a functional group in organic chemistry with the formula -OCH3. It consists of an oxygen atom bonded to a methyl group (CH3). The methoxy group is an ether, contributing to the molecule's polarity and influencing its solubility and reactivity. It is commonly used in organic synthesis to modify the electronic properties of aromatic rings, and in medicinal chemistry to enhance the pharmacokinetic properties of drugs, such as solubility, metabolic stability, and bioavailability.",
            "The side-chain hydroxyls group refers to a hydroxyl group (-OH) attached to the side chain of an amino acid or other organic molecule. This group is highly polar and can form hydrogen bonds, significantly affecting the molecule's solubility, reactivity, and interactions with other molecules. In proteins, side-chain hydroxyl groups are found in amino acids like serine, threonine, and tyrosine, where they play crucial roles in enzyme catalysis, signaling, and structural stability by participating in hydrogen bonding and phosphorylation reactions.",
            "The side-chain aldehydes or ketones group refers to aldehyde (-CHO) or ketone (C=O) functional groups attached to the side chains of organic molecules, including amino acids and other biomolecules. These groups are highly reactive due to the presence of a carbonyl group (C=O), which makes them susceptible to nucleophilic addition reactions. In biochemistry, side-chain aldehydes and ketones can play roles in metabolic pathways, enzyme catalysis, and signaling. For example, the aldehyde group in pyridoxal phosphate (a form of vitamin B6) is crucial for its function as a coenzyme in amino acid metabolism.",
            "The primary amines group consists of a nitrogen atom bonded to one alkyl or aryl group and two hydrogen atoms, with the general formula R-NH2. This group is highly reactive and can form hydrogen bonds, significantly affecting the molecule's solubility and chemical behavior. Primary amines are found in many biological molecules, including amino acids, neurotransmitters, and vitamins, playing crucial roles in protein structure, enzyme activity, and cell signaling due to their ability to participate in hydrogen bonding, nucleophilic reactions, and as proton acceptors or donors.",
            "???",
            "The nitriles group, also known as cyano group, has the formula -C≡N, consisting of a carbon atom triple-bonded to a nitrogen atom. This group is highly polar and electron-withdrawing, significantly affecting the reactivity and physical properties of the molecule it is attached to. Nitriles are found in various natural products, pharmaceuticals, and synthetic intermediates, playing important roles in organic synthesis due to their ability to undergo numerous chemical transformations, such as hydrolysis to carboxylic acids and reduction to amines.",
                        ]
        fName=os.path.join(RDConfig.RDDataDir,'FunctionalGroups.txt')
        from rdkit.Chem import FragmentCatalog
        fparams = FragmentCatalog.FragCatParams(1,6,fName)
        fcat=FragmentCatalog.FragCatalog(fparams)
        fcgen=FragmentCatalog.FragCatGenerator()
        m = Chem.MolFromSmiles(smiles)
        fcgen.AddFragsFromMol(m,fcat)
        num_entries=fcat.GetNumEntries()
        # print(smiles)
        fg_prompt=f"This molecule has "
        # if num_part is not None:
        #     fg_prompt += f"{num_part[0]} atoms and {num_part[1]} motifs, along with "
        added= False
        if num_entries:
            # print(list(fcat.GetEntryFuncGroupIds(num_entries-1)))
            ans_list = list(fcat.GetEntryFuncGroupIds(num_entries-1))
            for fg_id in np.unique(ans_list):
                # print(fg_id,len(fgroup_names))
                fg_name = fgroup_names[fg_id]

                if fg_name != "???":
                    fg_prompt += f"{ans_list.count(fg_id)} {fg_name} group, and "
                    added=True
        if added:
            fg_prompt = fg_prompt[:-len(", and ")]+"."
            if not self.data_args.add_fg_prompt_neg2:
                for fg_id in np.unique(ans_list):
                    fg_prompt += fgroup_intro[fg_id]
        else:
            fg_prompt += "0 functional groups."
        return fg_prompt
    def get_fg_prompt_neg(self,smiles,num_part=None,all=False):
        fgroup_names = ['methyl amide','carboxylic acids','carbonyl methyl ester','terminal aldehyde','amide','carbonyl methyl','isocyanate','isothiocyanate',
                        'nitro','nitroso','oximes','Imines','Imines','terminal azo','hydrazines','diazo','cyano',
                        'primary sulfonamide','methyl sulfonamide','sulfonic acid','methyl ester sulfonyl',
                        'methyl sulfonyl','sulfonyl chloride','methyl sulfinyl','methylthio','thiols','thiocarbonyls',
                        'halogens','t-butyl','trifluoromethyl','acetylenes','cyclopropyl',
                        'ethoxy','methoxy','side-chain hydroxyls','side-chain aldehydes or ketones','primary amines',
                        '???',
                        'nitriles']
        fName=os.path.join(RDConfig.RDDataDir,'FunctionalGroups.txt')
        from rdkit.Chem import FragmentCatalog
        fparams = FragmentCatalog.FragCatParams(1,6,fName)
        fcat=FragmentCatalog.FragCatalog(fparams)
        fcgen=FragmentCatalog.FragCatGenerator()
        m = Chem.MolFromSmiles(smiles)
        fcgen.AddFragsFromMol(m,fcat)
        num_entries=fcat.GetNumEntries()
        # print(smiles)
        fg_prompt=f"There are no "
        # if num_part is not None:
        #     fg_prompt += f"{num_part[0]} atoms and {num_part[1]} motifs, along with "
        # added= False
        ans = np.ones(len(fgroup_names))
        ans[-2] = 0
        if num_entries:
            # print(list(fcat.GetEntryFuncGroupIds(num_entries-1)))
            ans_list = list(fcat.GetEntryFuncGroupIds(num_entries-1))
            for aaa in ans_list:
                ans[aaa] = 0
        selected_names = []
        import random
        if all:
            # sample_len = int(ans.sum())
            for (fg_id,aaa) in enumerate(ans):
                if aaa==0:
                    continue
                fg_name = fgroup_names[fg_id]

                if fg_name != "???":
                    fg_prompt += fg_name+", or "
        else:
            sample_len = 4
            while len(selected_names)<sample_len:
                cur = random.choices(np.arange(len(fgroup_names)),weights=ans,k=sample_len)
                for cc in cur:
                    if fgroup_names[cc] in selected_names:
                        continue
                    else:
                        selected_names.append(fgroup_names[cc])
                        fg_prompt += fgroup_names[cc]+", or "
        fg_prompt = fg_prompt[:-len(", or ")]+" groups present in this molecule."
        # if added:
        #     fg_prompt += "functional groups."
        # else:
        #     fg_prompt += "0 functional groups."
        return fg_prompt
    def get_fg_prompt_bin(self,smiles,num_part=None):
        fgroup_names = ['methyl amide','carboxylic acids','carbonyl methyl ester','terminal aldehyde','amide','carbonyl methyl','isocyanate','isothiocyanate',
                        'nitro','nitroso','oximes','Imines','Imines','terminal azo','hydrazines','diazo','cyano',
                        'primary sulfonamide','methyl sulfonamide','sulfonic acid','methyl ester sulfonyl',
                        'methyl sulfonyl','sulfonyl chloride','methyl sulfinyl','methylthio','thiols','thiocarbonyls',
                        'halogens','t-butyl','trifluoromethyl','acetylenes','cyclopropyl',
                        'ethoxy','methoxy','side-chain hydroxyls','side-chain aldehydes or ketones','primary amines',
                        '???',
                        'nitriles']
        fName=os.path.join(RDConfig.RDDataDir,'FunctionalGroups.txt')
        from rdkit.Chem import FragmentCatalog
        fparams = FragmentCatalog.FragCatParams(1,6,fName)
        fcat=FragmentCatalog.FragCatalog(fparams)
        fcgen=FragmentCatalog.FragCatGenerator()
        m = Chem.MolFromSmiles(smiles)
        fcgen.AddFragsFromMol(m,fcat)
        num_entries=fcat.GetNumEntries()
        

        question_pools = [
            'Could you find a <fg> group in this molecule?',
            'Is there a <fg> group in this molecule?',
            'Is there any <fg> group present in this molecule?',
            'Does the molecule contain a <fg> group?',
        ]
        answer_pools = [[
            'Yes, there is a <fg> group in this molecule.',
            'Yes, the <fg> group is present in this molecule.',
            'Yes, there exists a <fg> group in this molecule.',
            'Yes, <fg> group exists in this molecule.',
            'Yes, the molecules contains a <fg> group.',
        ],[
            'No, there is no <fg> group in this molecule.',
            'No, the <fg> group is not present in this molecule.',
            'No, there does not exists a <fg> group in this molecule.',
            'No, <fg> group does not exist in this molecule.',
            'No, the molecules contains a <fg> group.',
        ]]
        question = random.choice(question_pools)
        sel_flag =  random.random() <= 0.5
        ans = np.ones(len(fgroup_names))-sel_flag
        # num_ids = list(fcat.GetEntryFuncGroupIds(num_entries-1))
        if num_entries:
            # print(list(fcat.GetEntryFuncGroupIds(num_entries-1)))
            ans_list = list(fcat.GetEntryFuncGroupIds(num_entries-1))
            for aaa in ans_list:
                ans[aaa] = sel_flag
        ans[-2] = 0
        if ans.sum()<1:
            sel_flag = 0
            ans = np.ones(len(fgroup_names))-sel_flag
        print(smiles,num_entries,ans,sel_flag,ans.sum())
        selected_names = []
        
        while len(selected_names)<1:
            cur = random.choices(np.arange(len(fgroup_names)),weights=ans,k=1)
            for cc in cur:
                if fgroup_names[cc] in selected_names:
                    continue
                else:
                    selected_names.append(fgroup_names[cc])
        answer = random.choice(answer_pools[sel_flag])
        question = question.replace("<fg>",fgroup_names[cc])
        answer = answer.replace("<fg>",fgroup_names[cc])
        return question, answer
