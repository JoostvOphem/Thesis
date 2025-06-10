from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np

from data_utils import get_dataset


DATASET = "Ghostbusters_all"  # options: "Ghostbusters_all", "gpt_writing", "monolingual_davinci", "GPT2"
WANDB_ENABLED = True
if WANDB_ENABLED:
    import wandb

    run = wandb.init(
        project="30-05-comparisons",
        name=f'SSL_{DATASET}'
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

# def normalize_embeddings(embeddings):
#     return  embeddings / torch.norm(embeddings, dim=1, keepdim=True)
#     # mean = embeddings.mean(dim=0, keepdim=True)
#     # std = embeddings.std(dim=0, keepdim=True)
#     # normalized_embeddings = (embeddings - mean) / (std + 1e-8)
#     # return normalized_embeddings

def load_path_in_tensor(path):
    return torch.tensor(np.array(torch.load(path)))

train_data, train_labels, val_data, val_labels, test_data, test_labels = get_dataset(DATASET)

print(type(train_data))

train_data = torch.nn.functional.normalize(train_data)
val_data = torch.nn.functional.normalize(val_data)
test_data = torch.nn.functional.normalize(test_data)

print(type(train_data))

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

    # Get max probabilities and predicted labels
    max_probs, predicted_labels = torch.max(pred_probs, dim=1)

    # Filter for high-confidence predictions (>= 0.8)
    confident_mask = max_probs >= 0.99
    num_confident = confident_mask.sum().item()
    pseudo_labels = predicted_labels[confident_mask].unsqueeze(1)
    data_pseudo_labeled = unsupervised_train_data[confident_mask]
    
    # Combine labeled and pseudo-labeled data
    data_train_combined = torch.concatenate([supervised_train_data, data_pseudo_labeled], axis=0)
    labels_train_combined = torch.concatenate([supervised_train_labels, pseudo_labels], axis=0)
    # ---------------

    # Train model again with both labeled and pseudo-labeled data
    model.fit(data_train_combined, labels_train_combined, epochs=10, validation_data=validation_set)

    evaluation_test = model.evaluate(test_data, test_labels)
    evaluation_train = model.evaluate(supervised_train_data, supervised_train_labels)

    if WANDB_ENABLED:
        wandb.log({
            "test_loss": evaluation_test[0],
            "test_accuracy": evaluation_test[1],
            "train_loss": evaluation_train[0],
            "train_accuracy": evaluation_train[1],
            "percentage of pseudolabeled items": num_confident / len(unsupervised_train_data),
            "epoch": i
        })