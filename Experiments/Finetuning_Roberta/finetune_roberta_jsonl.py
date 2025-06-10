import torch
import torch.nn as nn
import os
import pandas as pd
import random

MODEL_TO_CHANGE = 'roberta-SemEval'
AMT_OF_TEXTS = 1000
path_to_jsonl = 'Datasets/SemEval/SemEval2024-M4/SubtaskA/subtaskA_train_monolingual.jsonl'

WANDB_ENABLED = True
if WANDB_ENABLED:
    import wandb
    run = wandb.init(
        project="experiments-9-06",
        name=f"finetune-{MODEL_TO_CHANGE}-{AMT_OF_TEXTS}",
    )

def prepare_texts_labels(path_to_jsonl):
    total_df = pd.read_json(path_or_buf=path_to_jsonl, lines=True)


    def drop_nan(df, col):
        indices = df[df[col].isna()].index
        df = df.drop(indices)
        return df
    
    for col in total_df:
        drop_nan(total_df, col)
    
    total_df = total_df.sample(frac=1).reset_index(drop=True)
    
    texts = total_df['text'].tolist()
    labels = total_df['label'].tolist()
    
    # # clean nan values because not all loaded data necessarily have 'Human' and 'AI' columns
    # total_df = drop_nan(total_df, 'Human')
    # total_df = drop_nan(total_df, 'AI')

    # shuffle texts
    indices = random.sample(range(len(labels)), len(labels))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    return texts, labels


texts, labels = prepare_texts_labels(path_to_jsonl)

# print("percentage human", labels.count(1) / len(labels))


texts = texts[:AMT_OF_TEXTS]
labels = labels[:AMT_OF_TEXTS]

labels = torch.tensor(labels, dtype=torch.float32)

### ------------------ ###
# Load the RoBERTa model and tokenizer

class RobertaWithClassifier(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(768, 2),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, 768]
        return self.classifier(pooled_output).squeeze()

from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaModel, RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True) # TODO: change to 8

model = RobertaWithClassifier()
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCELoss()

best_loss = float('inf')

num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, batch_labels = batch
        optimizer.zero_grad()
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    
    if WANDB_ENABLED:
        wandb.log({"epoch": epoch, 
                   "loss": avg_loss,
                   "accuracy": (predictions.round() == batch_labels).float().mean().item()})

    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"Saving best model at epoch {epoch}, loss: {best_loss:.4f}")
        model.roberta.save_pretrained(f"robertas/{MODEL_TO_CHANGE}")
        tokenizer.save_pretrained(f"robertas/{MODEL_TO_CHANGE}")
        torch.save(model.classifier.state_dict(), f"robertas/{MODEL_TO_CHANGE}/best_classifier.pt")