import torch
import numpy as np
import json

from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn


WANDB_ENABLED = False
if WANDB_ENABLED:
    import wandb

    run = wandb.init(
        project="SSL_Roberta_compare",
        name='Transfor_to_torch'
    )

# data = torch.tensor(np.array(torch.load("subset_embeddings.npy")))
# labels = torch.load("subset_labels.npy")

# def ss_model(loss_function = "sparse_categorical_crossentropy",
#              optimizer = "adamw",
#              metrics=["accuracy"],
#              epochs=30,
#              layers = [tf.keras.layers.Flatten(input_shape=[768,1], name="inputLayer"),
#                        tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
#                        tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
#                        tf.keras.layers.Dense(10, activation="relu", name="hiddenlayer3"),
#                        tf.keras.layers.Dense(2, activation="softmax", name="outputlayer")]
#             ):

#     model = tf.keras.models.Sequential(layers)
    
#     model.compile(loss=loss_function,
#                  optimizer=optimizer,
#                  metrics=metrics)

#     return model


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(768, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

# train, test, val = 0.6, 0.3, 0.1 # percentages
supervised_percentage = 0.1

def label_data(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data = json.loads(line)  # Parse each line as JSON
                labels.append(data["label"])
    
    # Convert to tensor
    return torch.tensor(labels, dtype=torch.float)


# train_data = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_train.jsonl")))
# train_labels = label_data("Datasets/Ghostbusters_standardized/gpt_writing_train.jsonl")

# val_data = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_val.jsonl")))
# val_labels = label_data("Datasets/Ghostbusters_standardized/gpt_writing_val.jsonl")

# test_data = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_test.jsonl")))
# test_labels = label_data("Datasets/Ghostbusters_standardized/gpt_writing_test.jsonl")

amt_to_test = 2000

def normalize_embeddings(embeddings):
    return  embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    # mean = embeddings.mean(dim=0, keepdim=True)
    # std = embeddings.std(dim=0, keepdim=True)
    # normalized_embeddings = (embeddings - mean) / (std + 1e-8)
    # return normalized_embeddings

train_data = torch.tensor(np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_train.jsonl")))[:amt_to_test]
train_data = normalize_embeddings(train_data)
train_labels = label_data("Datasets/SemEval_standardized/monolingual/monolingual_davinci_train.jsonl")[:amt_to_test]

val_data = torch.tensor(np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_val.jsonl")))[:amt_to_test]
val_data = normalize_embeddings(val_data)
val_labels = label_data("Datasets/SemEval_standardized/monolingual/monolingual_davinci_val.jsonl")[:amt_to_test]

test_data = torch.tensor(np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_test.jsonl")))[:amt_to_test]
test_data = normalize_embeddings(test_data)
test_labels = label_data("Datasets/SemEval_standardized/monolingual/monolingual_davinci_test.jsonl")[:amt_to_test]


supervised_train_data = train_data[:int(supervised_percentage * len(train_data))]
supervised_train_labels = train_labels[:int(supervised_percentage * len(train_labels))]
unsupervised_train_data = train_data[int(supervised_percentage * len(train_data)):]
unsupervised_train_labels = train_labels[int(supervised_percentage * len(train_labels)):]

# # shuffle the data
# def shuffle_data(data, labels):
#     indices = torch.randperm(len(data))
#     return data[indices], labels[indices]

# supervised_train_data, supervised_train_labels = shuffle_data(supervised_train_data, supervised_train_labels)
# unsupervised_train_data, unsupervised_train_labels = shuffle_data(unsupervised_train_data, unsupervised_train_labels)
# val_data, val_labels = shuffle_data(val_data, val_labels)
# test_data, test_labels = shuffle_data(test_data, test_labels)

validation_set = (val_data, val_labels)


model = MLPClassifier()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 20
inner_epochs = 20  # Separate the inner training epochs

for i in range(num_epochs):
    print(f"\n=== Pseudo-labeling Iteration {i+1}/{num_epochs} ===")
    
    # Generate pseudo labels
    model.eval()
    with torch.no_grad():
        # Get prediction probabilities for unsupervised data
        raw_outputs = model(unsupervised_train_data.float())
        pred_probs = torch.sigmoid(raw_outputs).squeeze()  # Convert to probabilities
        
        # Create binary predictions
        predicted_labels = (pred_probs > 0.5).float()
        
        # Filter for high-confidence predictions
        # For binary classification, confidence is distance from 0.5
        confidence = torch.abs(pred_probs - 0.5)
        confident_mask = confidence >= 0.45  # High confidence (>= 0.95 or <= 0.05 probability)
        
        if confident_mask.sum() > 0:
            pseudo_labels = predicted_labels[confident_mask]
            data_pseudo_labeled = unsupervised_train_data[confident_mask]

            print(pseudo_labels.shape)
            print(pseudo_labels)
            
            # Combine labeled and pseudo-labeled data
            data_train_combined = torch.cat([supervised_train_data, data_pseudo_labeled], dim=0)
            labels_train_combined = torch.cat([supervised_train_labels, pseudo_labels], dim=0)
            
            print(f"Added {confident_mask.sum().item()} pseudo-labeled samples")
        else:
            # If no confident predictions, just use supervised data
            data_train_combined = supervised_train_data
            labels_train_combined = supervised_train_labels
            print("No confident pseudo-labels found, using only supervised data")

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(data_train_combined, labels_train_combined), 
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data, val_labels), 
        batch_size=32, shuffle=False
    )

    # Train model with combined data
    model.train()
    train_loss = 0
    for epoch in range(inner_epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.float()
            y_batch = y_batch.float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

    # Evaluate on validation set
    model.eval()
    correct, total, val_loss = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.float()
            y_batch = y_batch.float()
            
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            val_loss += loss.item()
            
            # Convert outputs to predictions
            preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()
            
            # Handle single sample case
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if y_batch.dim() == 0:
                y_batch = y_batch.unsqueeze(0)
                
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Predictions sample: {all_preds[:10]}")
    print(f"Labels sample: {all_labels[:10]}")
    
    # Optional: Early stopping if accuracy is very low
    if i > 3 and accuracy < 0.6:  # If still not learning after a few iterations
        print("Model not learning well, consider:")
        print("1. Checking data preprocessing")
        print("2. Adjusting learning rate")
        print("3. Verifying label distribution")
        break


# # model = ss_model()

# model = MLPClassifier()
# # criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


# num_epochs = 20
# for i in range(num_epochs):
#     # ---------------
#     # generate pseudo labels

#     # Get prediction probabilities
#     pred_probs = torch.tensor(model.forward(unsupervised_train_data))  # shape: (N, num_classes)

#     # Get max probabilities and predicted labels
#     max_probs, predicted_labels = torch.max(pred_probs, dim=1)

#     # Filter for high-confidence predictions
#     confident_mask = max_probs >= 0.95
#     pseudo_labels = predicted_labels[confident_mask]
#     data_pseudo_labeled = unsupervised_train_data[confident_mask]
    
#     # Combine labeled and pseudo-labeled data
#     data_train_combined = torch.concatenate([supervised_train_data, data_pseudo_labeled], axis=0)
#     labels_train_combined = torch.concatenate([supervised_train_labels, pseudo_labels], axis=0)

#     train_loader = DataLoader(TensorDataset(data_train_combined, labels_train_combined), batch_size=32, shuffle=True)
#     val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=32, shuffle=True)
#     # ---------------

#     # Train model again with both labeled and pseudo-labeled data
#     # model.fit(data_train_combined, labels_train_combined, epochs=10, validation_data=validation_set)
#     for epoch in range(num_epochs):
    
#         model.train()

#         for x_batch, y_batch in train_loader:
#             x_batch = x_batch.float()
#             y_batch = y_batch.float().unsqueeze(1)
#             optimizer.zero_grad()
#             outputs = model(x_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()

#     # evaluation = model.evaluate(test_data, test_labels)

#     model.eval()
#     correct, total, val_loss = 0, 0, 0
#     with torch.no_grad():
#         for x_batch, y_batch in val_loader:
#             x_batch = x_batch.float()
#             y_batch = y_batch.float().unsqueeze(1)
#             outputs = model(x_batch)
#             loss = criterion(outputs, y_batch.float())
#             val_loss += loss.item()
#             preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()            
#             print(preds.shape, y_batch.shape)
#             print(preds)
#             print(y_batch)
#             # correct += (preds == y_batch).sum().item()
#             for pred, label in zip(preds, y_batch):
#                 if pred == label:
#                     correct += 1
#             print(correct, total)
#             total += y_batch.size(0)
#     accuracy = correct / total

#     print(f"Epoch {i+1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}")
#     # print(model.metrics_names)
#     # print(evaluation)

#     if WANDB_ENABLED:
#         wandb.log({
#             "loss": val_loss / len(val_loader),
#             "accuracy": accuracy,
#             "epoch": i
#         })