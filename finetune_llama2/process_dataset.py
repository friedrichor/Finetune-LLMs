import os
import sys
from itertools import chain
import datasets

from transformers import LlamaTokenizer

model_id = "/datas/huggingface/Llama-2-7b-hf/"
tokenizer = LlamaTokenizer.from_pretrained(model_id)

dataset = datasets.load_dataset("samsum", split="train")
print(dataset)
# Dataset({
#     features: ['id', 'dialogue', 'summary'],
#     num_rows: 14732
# })
print(dataset[0])
# {'id': '13818513', 
#  'dialogue': "Amanda: I baked  cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)", 
#  'summary': 'Amanda baked cookies and will bring Jerry some tomorrow.'}
print("="*50)

prompt = (
    f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
)

def apply_prompt_template(sample):
    return {
        "text": prompt.format(
            dialog=sample["dialogue"],
            summary=sample["summary"],
            eos_token=tokenizer.eos_token,
        )
    }

dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
print(dataset)
# Dataset({
#     features: ['text'],
#     num_rows: 14732
# })
print(dataset[0])
# {'text': "Summarize this dialog:\nAmanda: I baked  cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)\n---\nSummary:\nAmanda baked cookies and will bring Jerry some tomorrow.</s>"}
print("="*50)

dataset = dataset.map(
    lambda sample: tokenizer(sample["text"]),
    batched=True,
    remove_columns=list(dataset.features),
)
print(dataset)
# Dataset({
#     features: ['input_ids', 'attention_mask'],
#     num_rows: 14732
# })
print(dataset[:2])
# {'input_ids': [[1, 6991, 3034, 675, 445, 7928, 29901, 13, 6833, 5863, 29901, 306, 289, 12535, 29871, 21046, 29889, 1938, 366, 864, 777, 29973, 30004, 13, 29967, 261, 719, 29901, 18585, 29991, 30004, 13, 6833, 5863, 29901, 306, 29915, 645, 6963, 366, 6454, 22396, 15626, 13, 5634, 13, 26289, 29901, 13, 6833, 5863, 289, 12535, 21046, 322, 674, 6963, 23052, 777, 6454, 22396, 29889, 2], 
#                [1, 6991, 3034, 675, 445, 7928, 29901, 13, 29949, 17843, 423, 29901, 11644, 526, 366, 28931, 363, 297, 445, 8271, 29973, 6756, 13, 29949, 14381, 29901, 10895, 1338, 408, 2337, 22993, 13, 29949, 17843, 423, 29901, 2191, 2086, 6824, 30004, 13, 29949, 14381, 29901, 7027, 13, 5634, 13, 26289, 29901, 13, 29949, 17843, 423, 322, 19802, 631, 526, 28931, 363, 7866, 1338, 297, 445, 8271, 29889, 29871, 2]], 
# 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

print("="*50)


class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}
        
    def __call__(self, batch):
        # batch = {"input_ids":[[], [], [], ...], "attention_mask":[[], [], [], ...]} (every 1000 items, not dataset)
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:  # resegmentation
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            # So far, the above operation has resulted in a concat of all the data,
            # and then cutting it up into several `chunk_size`` long sentences.
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result
    
dataset = dataset.map(Concatenator(), batched=True)

print(dataset)
# Dataset({
#     features: ['input_ids', 'attention_mask', 'labels'],
#     num_rows: 1555
# })
print(dataset[:2]) 
# {'input_ids': [[chunk_size items], [chunk_size items]],
#  'attention_mask': [[chunk_size items], [chunk_size items]],
#  'labels': [[chunk_size items], [chunk_size items]]}
print(tokenizer.decode(dataset[0]['input_ids']))
print("="*50)

dataset_name = "samsum_for_llm"
# Save dataset to disk
dataset.save_to_disk(os.path.join(sys.path[0], f"datasets/{dataset_name}"))
