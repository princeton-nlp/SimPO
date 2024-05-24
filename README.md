# Simple Preference Optimization (SimPO)

This repository contains the code and released models for our paper [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734). We propose a simpler and more effective preference optimization algorithm than DPO (Direct Preference Optimization) without using a reference model. SimPO outperforms DPO and its latest variants across AlpacaEval 2, MT-Bench, and Arena-Hard benchmarks under various settings.

<img src="./SimPO.png" width="1000px"></img>

## 🔗 Quick Links
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](#simple-preference-optimization-simpo)
  - [Released Models](#released-models)
  - [Install Requirements](#install-requirements)
  - [Training scripts](#training-scripts)
  - [Evaluation](#evaluation)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Released Models
Below is the full list of models that we evaluate in our preprint.

| models                       |                                                                                                           | AE2 LC | AE2 WR |  AH  |
|------------------------------|-----------------------------------------------------------------------------------------------------------|:------:|:------:|:----:|
| Mistral Base 7B SFT          | [alignment-handbook/zephyr-7b-sft-full](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full)     |   8.4  |   6.2  |  1.3 |
| Mistral Base 7B DPO (Zephyr) | [princeton-nlp/Mistral-7B-Base-SFT-DPO](https://huggingface.co/princeton-nlp/Mistral-7B-Base-SFT-DPO)     |  15.1  |  12.5  | 10.4 |
| Mistral Base 7B IPO          | [princeton-nlp/Mistral-7B-Base-SFT-IPO](https://huggingface.co/princeton-nlp/Mistral-7B-Base-SFT-IPO)     |  11.8  |   9.4  |  7.5 |
| Mistral Base 7B KTO          | [princeton-nlp/Mistral-7B-Base-SFT-KTO](https://huggingface.co/princeton-nlp/Mistral-7B-Base-SFT-KTO)     |  13.1  |   9.1  |  5.6 |
| Mistral Base 7B ORPO         | [kaist-ai/mistral-orpo-beta](https://huggingface.co/kaist-ai/mistral-orpo-beta)                           |  14.7  |  12.2  |  7.0 |
| Mistral Base 7B R-DPO        | [princeton-nlp/Mistral-7B-Base-SFT-RDPO](https://huggingface.co/princeton-nlp/Mistral-7B-Base-SFT-RDPO)   |  17.4  |  12.8  |  9.9 |
| Mistral Base 7B SimPO        | [princeton-nlp/Mistral-7B-Base-SFT-SimPO](https://huggingface.co/princeton-nlp/Mistral-7B-Base-SFT-SimPO) |  21.4  |  20.8  | 16.6 |
| Mistral Instruct 7B SFT      | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)           |  17.1  |  14.7  | 12.6 |
| Mistral Instruct 7B DPO      | [princeton-nlp/Mistral-7B-Instruct-DPO](https://huggingface.co/princeton-nlp/Mistral-7B-Instruct-DPO)     |  26.8  |  24.9  | 16.3 |
| Mistral Instruct 7B IPO      | [princeton-nlp/Mistral-7B-Instruct-IPO](https://huggingface.co/princeton-nlp/Mistral-7B-Instruct-IPO)     |  20.3  |  20.3  | 16.2 |
| Mistral Instruct 7B KTO      | [princeton-nlp/Mistral-7B-Instruct-KTO](https://huggingface.co/princeton-nlp/Mistral-7B-Instruct-KTO)     |  24.5  |  23.6  | 17.9 |
| Mistral Instruct 7B ORPO     | [princeton-nlp/Mistral-7B-Instruct-ORPO](https://huggingface.co/princeton-nlp/Mistral-7B-Instruct-ORPO)   |  24.5  |  24.9  | 20.8 |
| Mistral Instruct 7B R-DPO    | [princeton-nlp/Mistral-7B-Instruct-RDPO](https://huggingface.co/princeton-nlp/Mistral-7B-Instruct-RDPO)   |  27.3  |  24.5  | 16.1 |
| Mistral Instruct 7B SimPO    | [princeton-nlp/Mistral-7B-Instruct-SimPO](https://huggingface.co/princeton-nlp/Mistral-7B-Instruct-SimPO) |  32.1  |  34.8  | 21.0 |
| Llama3 Base 8B SFT           | [princeton-nlp/Llama-3-Base-8B-SFT](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT)             |   6.2  |   4.6  |  3.3 |
| Llama3 Base 8B DPO           | [princeton-nlp/Llama-3-Base-8B-DPO](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-DPO)             |  18.2  |  15.5  | 15.9 |
| Llama3 Base 8B IPO           | [princeton-nlp/Llama-3-Base-8B-IPO](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-IPO)             |  14.4  |  14.2  | 17.8 |
| Llama3 Base 8B KTO           | [princeton-nlp/Llama-3-Base-8B-KTO](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-KTO)             |  14.2  |  12.4  | 12.5 |
| Llama3 Base 8B ORPO          | [princeton-nlp/Llama-3-Base-8B-ORPO](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-ORPO)           |  12.2  |  10.6  | 10.8 |
| Llama3 Base 8B R-DPO         | [princeton-nlp/Llama-3-Base-8B-RDPO](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-RDPO)           |  17.6  |  14.4  | 17.2 |
| Llama3 Base 8B SimPO         | [princeton-nlp/Llama-3-Base-8B-SimPO](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SimPO)         |  22.0  |  20.3  | 23.4 |
| Llama3 Instruct 8B SFT       | [meta-llama/Meta-Llama-3-Instruct-8B](https://huggingface.co/meta-llama/Meta-Llama-3-Instruct-8B)         |  26.0  |  25.3  | 22.3 |
| Llama3 Instruct 8B DPO       | [princeton-nlp/Llama-3-Instruct-8B-DPO](https://huggingface.co/princeton-nlp/Llama-3-Instruct-8B-DPO)     |  40.3  |  37.9  | 32.6 |
| Llama3 Instruct 8B IPO       | [princeton-nlp/Llama-3-Instruct-8B-IPO](https://huggingface.co/princeton-nlp/Llama-3-Instruct-8B-IPO)     |  35.6  |  35.6  | 30.5 |
| Llama3 Instruct 8B KTO       | [princeton-nlp/Llama-3-Instruct-8B-KTO](https://huggingface.co/princeton-nlp/Llama-3-Instruct-8B-KTO)     |  33.1  |  31.8  | 26.4 |
| Llama3 Instruct 8B ORPO      | [princeton-nlp/Llama-3-Instruct-8B-ORPO](https://huggingface.co/princeton-nlp/Llama-3-Instruct-8B-ORPO)   |  28.5  |  27.4  | 25.8 |
| Llama3 Instruct 8B R-DPO     | [princeton-nlp/Llama-3-Instruct-8B-RDPO](https://huggingface.co/princeton-nlp/Llama-3-Instruct-8B-RDPO)   |  41.1  |  37.8  | 33.1 |
| Llama3 Instruct 8B SimPO     | [princeton-nlp/Llama-3-Instruct-8B-SimPO](https://huggingface.co/princeton-nlp/Llama-3-Instruct-8B-SimPO) |  44.7  |  40.5  | 33.8 |

## Install Requirements

Our codebase is built upon the [alignment-handbook repo](https://github.com/huggingface/alignment-handbook). The following steps will guide you through the installation process.

First, create a Python virtual environment using e.g. Conda:
```shell
conda create -n handbook python=3.10 && conda activate handbook
```

Next, install PyTorch `v2.2.2`. Since this is hardware-dependent, we
direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/locally/).

You can then install the remaining package dependencies of [alignment-handbook](https://github.com/huggingface/alignment-handbook) as follows:

```shell
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn --no-build-isolation
```

## Training Scripts

We provide four training config files for the four training setups reported in our paper:

* Mistral-Base:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/mistral-7b-base-simpo.yaml
```
* Mistral-Instruct:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/mistral-7b-instruct-simpo.yaml
```
* Llama3-Base:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-base-simpo.yaml
```
* Llama3-Instruct:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo.yaml
```

## Evaluation

We follow the official implementation for evaluation on AlpacaEval 2, Arena-Hard, and MT-Bench, as follows:

* AlpacaEval 2: Please refer to the [AlpacaEval repo](https://github.com/tatsu-lab/alpaca_eval) for evaluation.

* Arena-Hard: Please refer to to the [Arena-Hard-Auto repo](https://github.com/lm-sys/arena-hard-auto) for evaluation.

* MT-Bench: Please refer to the [FastChat repo](https://github.com/lm-sys/FastChat) for evaluation.

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Yu (yumeng5@virginia.edu). If you encounter any problems when using the code, or want to report a bug, feel free to open an issue! Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{meng2024simpo,
  title={{SimPO}: Simple Preference Optimization with a Reference-Free Reward},
  author={Meng, Yu and Xia, Mengzhou and Chen, Danqi},
  year={2024}
}
```
