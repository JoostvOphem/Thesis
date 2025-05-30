import torch
import torch.nn as nn
import os
from transformers import RobertaModel, RobertaTokenizer

type_name = "semeval_monolingual_300"

class Classifier(nn.Module):
    def __init__(self, hidden_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings):
        return self.sigmoid(self.linear(embeddings))

def find_files(dataset_dir):
    paths = [
        os.path.join(dataset_dir, path)
        for path in os.listdir(dataset_dir) 
    ]
    return paths

MODEL_NAME = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
roberta = RobertaModel.from_pretrained(MODEL_NAME)

classifier = Classifier(hidden_dim=768)

# WARNING: THIS IS THE LOCATION IN SNELLIUS, NOT LOCALLY
state_dict = torch.load(f"best_roberta_{type_name}/best_classifier.pt")
remove_prefix = '0'
state_dict = {'linear'+k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}

classifier.load_state_dict(state_dict)
classifier.eval()

def classify_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = roberta(**inputs)
        pooled = outputs.pooler_output  # shape: [1, 768]
        prediction = classifier(pooled)
    return prediction.item()

def embed_layer(text, layer=-2): # layer kinda arbitrarily chosen for now
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = roberta(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        cls_embedding = hidden_states[layer][:, 0, :]  # use the [CLS] token
    return cls_embedding

### ------------------ ###
# Embed the data
import pandas as pd

data_version = "Ghostbusters_all"  # change this to the desired dataset version
AMT_TO_EMBED = 5000

if data_version == "gpt_writing":
    input_files = ["Datasets/Ghostbusters_standardized/gpt_writing_train.jsonl",
                   "Datasets/Ghostbusters_standardized/gpt_writing_test.jsonl",
                   "Datasets/Ghostbusters_standardized/gpt_writing_val.jsonl"]
    output_folder = "Datasets/Ghostbusters_standardized_embedded"

if data_version == "Ghostbusters_all":
    input_files = ["Datasets/Ghostbusters_standardized/ghostbusters_ALL_train.jsonl",
                   "Datasets/Ghostbusters_standardized/ghostbusters_ALL_test.jsonl",
                   "Datasets/Ghostbusters_standardized/ghostbusters_ALL_val.jsonl"]
    output_folder = "Datasets/Ghostbusters_standardized_embedded"

if data_version == "monolingual_davinci":
    input_files = ["Datasets/SemEval_standardized/monolingual/monolingual_davinci_train.jsonl",
                "Datasets/SemEval_standardized/monolingual/monolingual_davinci_test.jsonl",
                "Datasets/SemEval_standardized/monolingual/monolingual_davinci_val.jsonl"]
    output_folder = "Datasets/SemEval_standardized_embedded/monolingual"

if data_version == "monolingual_complete":
    input_files = ["Datasets/SemEval_standardized/monolingual/monolingual_complete_train.jsonl",
                "Datasets/SemEval_standardized/monolingual/monolingual_complete_test.jsonl",
                "Datasets/SemEval_standardized/monolingual/monolingual_complete_val.jsonl"]
    output_folder = "Datasets/SemEval_standardized_embedded/monolingual"

if data_version == "GPT2":
    input_files = ["Datasets/GPT2_standardized/gpt2_complete.jsonl",
                   "Datasets/GPT2_standardized/gpt2_test.jsonl",
                   "Datasets/GPT2_standardized/gpt2_val.jsonl",
                   "Datasets/GPT2_standardized/gpt2_train.jsonl"]
    output_folder = "Datasets/GPT2_standardized_embedded"

print("WARNING TO FUTURE ME: LAYER=-2")
for input_file in input_files:
    jsonObj = pd.read_json(path_or_buf=input_file, lines=True)
    jsonObj = jsonObj.reset_index(drop=True)

    embeddings = []
    labels = []

    for i, row in jsonObj.iterrows():
        if i % 100 == 0 and i > 0:
            print(f"Processing row {i} / {len(jsonObj)} of {input_file}")
        
        if i == AMT_TO_EMBED:
            break
        text = row['text']
        embedding = embed_layer(text)
        embeddings.append(embedding)

        label = row['label']
        labels.append(label)
        
    final_text_tensor = torch.stack(embeddings)
    final_label_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    file_ending = input_file.split("/")[-1]
    torch.save(final_text_tensor, output_folder + "/" + file_ending)
    torch.save(final_label_tensor, output_folder + "/" + file_ending.replace('.jsonl', '_labels.pt'))
    print(f"Saved {file_ending} to {output_folder}")