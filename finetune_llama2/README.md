
# Getting Start

## Hardware

&#11088; The minimum requirement is a 24GB GPU.

## Installation

### 1. Prepare the code and the environment

```
git clone https://github.com/friedrichor/Finetune-LLMs.git
cd Finetune-LLMs/finetune_llama2
conda create -n llama2 python==3.10
conda activate llama2
pip install -r requirements.txt
conda install -y cudatoolkit
```

### 2. Prepare the model weights

First, please request the model weights of LLaMA 2 from Meta AI, and then convert them to hf (hugging face) format. As for how to convert the model format you can refer to https://www.bilibili.com/video/BV16X4y1L7EG.

The final weights would be in a single folder in a structure similar to the following:
```
Llama-2-7b-hf
├── config.json
├── generation_config.json
...
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── pytorch_model-00001-of-00002.bin
├── pytorch_model-00002-of-00002.bin
├── pytorch_model.bin.index.json
...
```

## Test base model

Run the base model on an example input:
```
CUDA_VISIBLE_DEVICES=0 python inference_base.py
```

## Prepare the dataset

```
python process_dataset.py
```

## Fine-tuning LLaMA 2 

```
CUDA_VISIBLE_DEVICES=0 python process_dataset.py
```

This process takes about 1.5 hours.

## Test fine-tuned model

Run the fine-tuned model on an example input:
```
CUDA_VISIBLE_DEVICES=0 python inference_finetuned.py
```