import json
import os
import random
from transformers import RobertaModel, RobertaTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def get_standardized_dataset_names(dataset_name):
    if dataset_name.startswith("monolingual"):
        start = "Datasets/SemEval_standardized/monolingual"
    elif dataset_name.startswith("multilingual"):
        start = "Datasets/SemEval_standardized/multilingual"
    elif dataset_name.startswith("gpt2"):
        start = "Datasets/GPT2_standardized"
    else:
        start = "Datasets/Ghostbusters_standardized"
    
    return [
        f"{start}/{dataset_name}_complete.jsonl",
        f"{start}/{dataset_name}_test.jsonl",
        f"{start}/{dataset_name}_train.jsonl",
        f"{start}/{dataset_name}_val.jsonl"
    ]

class RobertaWithClassifier(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, 768]
        return self.classifier(pooled_output).squeeze()

def get_roberta_and_tokenizer(dataset_name):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta = RobertaModel.from_pretrained('roberta-base')

    roberta_state_dict = torch.load(f"robertas/{dataset_name}/best_roberta.pt")

    roberta.load_state_dict(roberta_state_dict)
    return roberta, tokenizer

def embed_layer(roberta,
                tokenizer,
                text, 
                layer=-2): # layer kinda arbitrarily chosen for now
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = roberta(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        cls_embedding = hidden_states[layer][:, 0, :]  # use the [CLS] token
    return cls_embedding

def get_input_output_files(dataset_name, other_output_folder=None):
    if other_output_folder is None:
        other_output_folder = dataset_name

    input_files = get_standardized_dataset_names(dataset_name)
    if other_output_folder.startswith("monolingual"):
        output_folder = f"Datasets/SemEval_standardized_embedded/monolingual/roberta_{other_output_folder}"
    elif other_output_folder.startswith("multilingual"):
        output_folder = f"Datasets/SemEval_standardized_embedded/multilingual/roberta_{other_output_folder}"
    elif other_output_folder.startswith("GPT2"):
        output_folder = f"Datasets/GPT2_standardized_embedded/roberta_{other_output_folder}"
    else:
        output_folder = f"Datasets/Ghostbusters_standardized_embedded/roberta_{other_output_folder}"
    
    return input_files, output_folder

def read_jsonl_dataset(input_file, additional_field_names=None):
    """
    Read a JSONL file back into lists of texts and labels.
    
    Args:
    input_file (str): Path to the input JSONL file
    
    Returns:
    tuple: Lists of texts and labels
    """
    texts = []
    labels = []

    output = {}
    if not additional_field_names is None:
        add_fields = {name: [] for name in additional_field_names}

    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            texts.append(sample['text'])
            labels.append(sample['label'])

            if not additional_field_names is None:
                for name in additional_field_names:
                    add_fields[name].append(sample[name])


    output['text'] = texts
    output['label'] = labels
    if not additional_field_names is None:
        for name in additional_field_names:
            output[name] = add_fields[name]
    
    return output