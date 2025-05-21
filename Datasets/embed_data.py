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

def embed_layer(text, layer=-2): # second to last layer, kinda arbitrarily chosen for now
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = roberta(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        cls_embedding = hidden_states[layer][:, 0, :]  # use the [CLS] token
    return cls_embedding

### ------------------ ###
# Embed the data
import pandas as pd

input_files = find_files("Datasets/SemEval_standardized/monolingual")
output_folder = "Datasets/SemEval_standardized_embedded/monolingual"

for input_file in input_files:
    jsonObj = pd.read_json(path_or_buf=input_file, lines=True)
    jsonObj = jsonObj.sample(frac=1).reset_index(drop=True)

    embeddings = []

    for i, row in jsonObj.iterrows():
        if i % 100 == 0 and i > 0:
            print(f"Processing row {i} / {len(jsonObj)} of {input_file}")
        
        if i == 2000:
            break
        text = row['text']
        embedding = embed_layer(text)
        embeddings.append(embedding)
        
    final_tensor = torch.stack(embeddings)

    file_ending = input_file.split("/")[-1]
    torch.save(final_tensor, output_folder + "/" + file_ending)
    print(f"Saved {file_ending} to {output_folder}")