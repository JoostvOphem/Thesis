from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np

from data_utils import get_dataset

DATASET1 = "Ghostbusters_all"  # options: "Ghostbusters_all", "gpt_writing", "monolingual_davinci", "GPT2", "SemEval_complete"
DATASET2 = "SemEval_complete"

WANDB_ENABLED = True
if WANDB_ENABLED:
    import wandb

    run = wandb.init(
        project="30-05-comparisons",
        name=f'SSL_joined_{DATASET1}_{DATASET2}'
    )

def ss_model(loss_function = "sparse_categorical_crossentropy",
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=0.0005,
                weight_decay=0.01),
            metrics=["accuracy"],
            layers=[tf.keras.layers.Flatten(input_shape=[768,1], name="inputLayer"),
            
            # First layer: Learn high-level patterns
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),  # Helps with gradient flow
            tf.keras.layers.Dropout(0.1),
            
            # Second layer: Refine patterns
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            
            # Third layer: Final feature extraction
            # tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            
            # Output layer
            tf.keras.layers.Dense(2, activation="softmax")
        ]
            
            ):

    model = tf.keras.models.Sequential(layers)
    
    model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=metrics)

    return model


supervised_percentage = 0.1

train_data_A, train_labels_A, val_data_A, val_labels_A, test_data_A, test_labels_A = get_dataset(DATASET1)
train_data_B, train_labels_B, val_data_B, val_labels_B, tetst_data_B, test_labels_B = get_dataset(DATASET2)

# append the two tensors
def append_and_shuffle(tensor1, tensor2, indices=None):
    combined = torch.cat((tensor1, tensor2), dim=0)

    if indices is None:
        indices = torch.randperm(combined.size(0))
    return combined[indices]

def get_indices(num_items):
    return torch.randperm(num_items).tolist()

num_train = len(train_data_A) + len(train_data_B)
indices = get_indices(num_train)
train_data = append_and_shuffle(train_data_A, train_data_B, indices)
train_labels = append_and_shuffle(train_labels_A, train_labels_B, indices)

num_val = len(val_data_A) + len(val_data_B)
indices = get_indices(num_val)
val_data = append_and_shuffle(val_data_A, val_data_B, indices)
val_labels = append_and_shuffle(val_labels_A, val_labels_B, indices)

num_test = len(test_data_A) + len(tetst_data_B)
indices = get_indices(num_test)
test_data = append_and_shuffle(test_data_A, tetst_data_B, indices)
test_labels = append_and_shuffle(test_labels_A, test_labels_B, indices)


# train_data_A = torch.nn.functional.normalize(train_data_A)
# val_data_A = torch.nn.functional.normalize(val_data_A)
# train_data_B = torch.nn.functional.normalize(train_data_B)

supervised_train_data = train_data[:int(supervised_percentage * len(train_data))]
supervised_train_labels = train_labels[:int(supervised_percentage * len(train_labels))]
unsupervised_train_data = train_data[int(supervised_percentage * len(train_data)):]
unsupervised_train_labels = train_labels[int(supervised_percentage * len(train_labels)):]

validation_set = (val_data, val_labels)

model = ss_model()

for i in range(20):
    # ---------------
    # generate pseudo labels

    # Get prediction probabilities
    pred_probs = torch.tensor(model.predict(unsupervised_train_data))  # shape: (N, num_classes)

    entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-8), dim=1)
    max_entropy = torch.log(torch.tensor(2.0))  # For binary classification
    confidence = 1 - (entropy / max_entropy)
    confident_mask = confidence >= 0.99

    # Get max probabilities and predicted labels
    max_probs, predicted_labels = torch.max(pred_probs, dim=1)

    # Filter for high-confidence predictions (>= 0.8)
    confident_mask = max_probs >= 0.999
    num_confident = confident_mask.sum().item()
    pseudo_labels = predicted_labels[confident_mask].unsqueeze(1)
    data_pseudo_labeled = unsupervised_train_data[confident_mask]
    
    # Combine labeled and pseudo-labeled data
    data_train_combined = torch.concatenate([supervised_train_data, data_pseudo_labeled], axis=0)
    labels_train_combined = torch.concatenate([supervised_train_labels, pseudo_labels], axis=0)
    # ---------------

    # Train model again with both labeled and pseudo-labeled data
    model.fit(data_train_combined, labels_train_combined, epochs=20, validation_data=validation_set)

    evaluation_test = model.evaluate(tetst_data_B, test_labels_B)
    evaluation_train = model.evaluate(test_data_A, test_labels_A)

    if WANDB_ENABLED:
        wandb.log({
            "test_loss": evaluation_test[0],
            "test_accuracy": evaluation_test[1],
            "train_loss": evaluation_train[0],
            "train_accuracy": evaluation_train[1],
            "percentage of pseudolabeled items": num_confident / len(unsupervised_train_data),
            "epoch": i
        })

run.finish()