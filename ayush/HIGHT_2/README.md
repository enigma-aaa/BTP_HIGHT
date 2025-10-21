<h1 align="center">HIGHT_2: Hierarchical Graph Tokenization for Graph-Language Alignment</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2406.14021"><img src="https://img.shields.io/badge/arXiv-2406.14021-b31b1b.svg" alt="Paper"></a>
    <a href="https://github.com/LFhase/HIGHT_2"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <!-- <a href="https://colab.research.google.com/drive/1t0_4BxEJ0XncyYvn_VyEQhxwNMvtSUNx?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> -->
    <a href="https://arxiv.org/abs/2406.14021"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=ICML%2725&color=blue"> </a>
    <a href="https://github.com/LFhase/HIGHT_2/blob/main/LICENSE"> <img alt="License" src="https://img.shields.io/github/license/LFhase/CIGA?color=blue"> </a>
    <!-- <a href="https://icml.cc/virtual/2024/poster/3455"> <img src="https://img.shields.io/badge/Video-grey?logo=Kuaishou&logoColor=white" alt="Video"></a> -->
    <!-- <a href="https://lfhase.win/files/slides/HIGHT_2.pdf"> <img src="https://img.shields.io/badge/Slides-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a> -->
   <!--  <a href="https://icml.cc/media/PosterPDFs/ICML%202022/a8acc28734d4fe90ea24353d901ae678.png"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a> -->
</p>

This repo contains the sample code for reproducing the results of our ICML 2025 paper: *[Hierarchical Graph Tokenization for Molecule-Language Alignment](https://arxiv.org/abs/2406.14021)*, which has also been presented at ICML 2024 workshop on [Foundation Models in the Wild](https://icml.cc/virtual/2024/workshop/29954). 😆😆😆

Updates:

- The customized datasets, including `HiPubChem` and `MotifHallu`, are [open-sourced](https://huggingface.co/datasets/lfhase/HIGHT_2).
- The model checkpoints are [open-sourced](https://huggingface.co/lfhase/HIGHT_2).


## Environment Setup

Mostly refer to LLaVA installation
1. Clone this repository and navigate to project folder

2. Install Package
- If you have any trouble install torch-geometric related packages, please refer to [guide-to-pyg-install](https://github.com/chao1224/GraphMVP#environments) for detailed instructions.
```Shell
conda create -n env_hight python=3.10 -y
conda activate env_hight
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# Install Graph related packages. We use torch-113 with CUDA-11.8, and pytorch_geometric=2.3.1 please change accordingly.
pip install -r requirements.txt
```

3. Install additional packages for training cases
```
pip install ninja
pip install flash-attn --no-build-isolation
```

## Weights

### Component Weights Download

- [] will be released soon!

## Dataset

- PubChem: Referring to [MoleculeSTM](https://github.com/chao1224/MoleculeSTM).
- Mol-Instructions: Referring to [Mol-Instructions](https://github.com/zjunlp/Mol-Instructions).


## Train
LLaVA training consists of two stages:

* **Stage 1: Alignment Pretraining.** Initial stage aligns molecules with text using a PubChem dataset of 330K pairs. Focuses on fine-tuning the alignment projector while keeping the graph encoder and LLM frozen to leverage pre-trained knowledge.
* **Stage 2: Task-specific Instruction Tuning.** Second stage targets compound property prediction, chemical reaction analysis, and molecule description generation. Utilizes task-specific instruction datasets and LoRA for LLM adaptation, retaining common-sense reasoning capabilities. Allows adaptable adaptors for specific needs or modular knowledge integration.

### Stage 1: Alignment Pretraining
See [pretrain.sh](scripts/pretrain.sh) for an example of how to run the pretraining stage.
- `$GRAPH_TOWER` can be chosen from `vqvae2` or `hvqvae2`.

### Stage 2: Task-specific Instruction Tuning
You can train all specific tasks combine together [finetune.sh](scripts/finetune.sh) or train them separately.


## Evaluation
See [Evaluation.md](Evaluation.md) for detailed instructions on how to evaluate the model.

## Misc

If you find our paper and repo useful, please cite our paper:

```bibtex
@inproceedings{chen2025hierarchical,
title={Hierarchical Graph Tokenization for Molecule-Language Alignment},
author={Yongqiang Chen and Quanming Yao and Juzheng Zhang and James Cheng and Yatao Bian},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=wpbNczwAwV}
}
```

We would like to acknowledge the contribution of [InstructMol](https://github.com/IDEA-XL/InstructMol) to the base codes.
