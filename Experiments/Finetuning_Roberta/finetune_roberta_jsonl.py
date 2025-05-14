import torch
import torch.nn as nn
import os
import pandas as pd
import random

WANDB_ENABLED = True
if WANDB_ENABLED:
    import wandb
    run = wandb.init(
        project="roberta-finetune",
        name="finetune-roberta-SemEval-postgithub",
    )

def prepare_texts_labels():
    
    def find_files(dataset_dir):
        paths = [
            os.path.join(dataset_dir, path)
            for path in os.listdir(dataset_dir) 
        ]
        return paths
    
    # # put the data from all files together
    # total_df = None
    # first=True
    # for file in find_files('/SemEval2024-M4/SubtaskA'):
    #     if file.startswith('~') or file.endswith('DS_Store'):
    #         continue
        
    #     if first:
    #         with pd.ExcelFile(file) as xls:  
    #             total_df = pd.read_excel(xls, "Sheet1")
    #             first=False
    #             continue
    #     with pd.ExcelFile(file) as xls:
    #         curr_df = pd.read_excel(xls, "Sheet1")
    
    #     total_df = pd.concat([total_df, curr_df], ignore_index=True)
    total_df = pd.read_json(path_or_buf='Datasets/SemEval/SemEval2024-M4/SubtaskA/subtaskA_train_monolingual.jsonl', lines=True)


    def drop_nan(df, col):
        indices = df[df[col].isna()].index
        df = df.drop(indices)
        return df
    
    for col in total_df:
        drop_nan(total_df, col)
    
    total_df = total_df.sample(frac=1).reset_index(drop=True)
    
    texts = total_df['text'].tolist()
    labels = total_df['label'].tolist()
    
    # # clean nan values because not all loaded data had 'Human' and 'AI' columns
    # total_df = drop_nan(total_df, 'Human')
    # total_df = drop_nan(total_df, 'AI')
    
    # texts = []
    # labels= []
    # for text in total_df['Human']:
    #     texts.append(text)
    #     labels.append(1)
    # for text in total_df['AI']:
    #     texts.append(text)
    #     labels.append(0)
    
    # # shuffle texts
    # indices = random.sample(range(len(labels)), len(labels))
    # texts = [texts[i] for i in indices]
    # labels = [labels[i] for i in indices]

    random.shuffle(texts)
    random.shuffle(labels)
    return texts, labels


texts, labels = prepare_texts_labels()

# print("percentage human", labels.count(1) / len(labels))


texts = texts[:100]
labels = labels[:100]

labels = torch.tensor(labels, dtype=torch.float32)
print("length of texts and labels:", len(texts), len(labels), torch.sum(labels))

### ------------------ ###
# Load the RoBERTa model and tokenizer

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

from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaModel, RobertaTokenizer

MODEL_NAME = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

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
        wandb.log({"epoch": epoch, "loss": avg_loss})

    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"Saving best model at epoch {epoch}, loss: {best_loss:.4f}")
        model.roberta.save_pretrained("best_roberta/")
        tokenizer.save_pretrained("best_roberta/")
        torch.save(model.classifier.state_dict(), "best_classifier.pt")

# from transformers import RobertaModel, RobertaTokenizer

# # Load a pretrained RoBERTa model and tokenizer
# MODEL_NAME = 'roberta-base'
# tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
# model = RobertaModel.from_pretrained(MODEL_NAME)

# # Define a classification head
# class Classifier(nn.Module):
#     def __init__(self, hidden_dim):
#         super(Classifier, self).__init__()
#         self.linear = nn.Linear(hidden_dim, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, embeddings):
#         return self.sigmoid(self.linear(embeddings))

# print("Instantiating the classifier")
# classifier = Classifier(hidden_dim=768)  # RoBERTa-base has a hidden size of 768

# def encode_texts(texts, tokenizer, model):
#     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.pooler_output  # Use pooled output for classification

# print("Embedding texts")
# embeddings = encode_texts(texts, tokenizer, model)

### ------------------ ###
# finetune Roberta

# WANDB_ENABLED = True
# if WANDB_ENABLED:
#     import wandb
#     run = wandb.init(
#         project="roberta-finetune"
#     )

# criterion = nn.BCELoss()
# optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001)

# # Training loop
# num_epochs = 1000
# best_loss = float('inf')

# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     predictions = classifier(embeddings).squeeze()
#     loss = criterion(predictions, labels)
#     loss.backward()
#     optimizer.step()

#     curr_loss = loss.item()
#     if curr_loss < best_loss:
#         best_loss = curr_loss
#         torch.save(classifier.state_dict(), 'best_model.pth')

#     if WANDB_ENABLED:
#         wandb.log({"epoch": epoch, "loss": curr_loss})
