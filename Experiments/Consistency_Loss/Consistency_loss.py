from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np
from transformers import RobertaModel, RobertaTokenizer


DATASET1 = "gpt_writing"  # Current options: "gpt_writing", "GPT2", "monolingual_davinci"
DATASET2 = "monolingual_davinci"

# model_name = 'roberta-base'
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# roberta = RobertaModel.from_pretrained(model_name)

# # classifier = Classifier(hidden_dim=768)

# state_dict = torch.load(f"best_roberta_monolingual_300/best_classifier.pt")
# roberta.load_state_dict(state_dict)

WANDB_ENABLED = True
if WANDB_ENABLED:
    import wandb

    run = wandb.init(
        project="CL_testing",
        name=f'27-05-{DATASET1}-{DATASET2}-joined_costum_loss'
    )

def difference_between(preds, C):
    """
    Computes the difference between the predictions and a constant C.
    This is used to track the consistency loss.
    """
    a, b = preds
    mid = (a + b) / 2
    return mid - C


def ss_model(loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
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
            # tf.keras.layers.Dropout(0.1),
            
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

def get_dataset(dataset_name):
    if dataset_name == "gpt_writing":
        train_data = load_path_in_tensor("Datasets/Ghostbusters_standardized_embedded/gpt_writing_train.jsonl")
        train_labels = load_path_in_tensor("Datasets/Ghostbusters_standardized_embedded/gpt_writing_train_labels.pt")

        val_data = load_path_in_tensor("Datasets/Ghostbusters_standardized_embedded/gpt_writing_val.jsonl")
        val_labels = load_path_in_tensor("Datasets/Ghostbusters_standardized_embedded/gpt_writing_val_labels.pt")

        test_data = load_path_in_tensor("Datasets/Ghostbusters_standardized_embedded/gpt_writing_test.jsonl")
        test_labels = load_path_in_tensor("Datasets/Ghostbusters_standardized_embedded/gpt_writing_test_labels.pt")
    elif dataset_name == "monolingual_davinci":
        train_data = load_path_in_tensor("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_train.jsonl")
        train_labels = load_path_in_tensor("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_train_labels.pt")

        val_data = load_path_in_tensor("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_val.jsonl")
        val_labels = load_path_in_tensor("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_val_labels.pt")

        test_data = load_path_in_tensor("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_test.jsonl")
        test_labels = load_path_in_tensor("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_test_labels.pt")
    elif dataset_name == "GPT2":
        train_data = load_path_in_tensor("Datasets/GPT2_standardized_embedded/gpt2_train.jsonl")
        train_labels = load_path_in_tensor("Datasets/GPT2_standardized_embedded/gpt2_train_labels.pt")

        val_data = load_path_in_tensor("Datasets/GPT2_standardized_embedded/gpt2_val.jsonl")
        val_labels = load_path_in_tensor("Datasets/GPT2_standardized_embedded/gpt2_val_labels.pt")

        test_data = load_path_in_tensor("Datasets/GPT2_standardized_embedded/gpt2_test.jsonl")
        test_labels = load_path_in_tensor("Datasets/GPT2_standardized_embedded/gpt2_test_labels.pt")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

train_data_A, train_labels_A, val_data_A, val_labels_A, _, _ = get_dataset(DATASET1)

if DATASET1 == DATASET2: # use validation set of dataset A for dataset B if they are the same to prevent duplicate data.
    _, _, _, _, train_data_B, train_labels_B = get_dataset(DATASET2)
else:
    train_data_B, train_labels_B, _, _, _, _ = get_dataset(DATASET2)


supervised_train_data = train_data_A[:int(supervised_percentage * len(train_data_A))]
supervised_train_labels = train_labels_A[:int(supervised_percentage * len(train_labels_A))]
unsupervised_train_data = train_data_B[int(supervised_percentage * len(train_data_B)):]
unsupervised_train_labels = train_labels_B[int(supervised_percentage * len(train_labels_B)):]

validation_set = (val_data_A, val_labels_A)

model = ss_model()
for i in range(50):
    # ---------------
    # generate pseudo labels

    # Get prediction probabilities
    pred_probs = torch.tensor(model.predict(unsupervised_train_data))  # shape: (N, num_classes)

    # Get max probabilities and predicted labels
    max_probs, predicted_labels = torch.max(pred_probs, dim=1)

    # Filter for high-confidence predictions (>= 0.8)
    confident_mask = max_probs >= 0.95
    num_confident = confident_mask.sum().item()
    pseudo_labels = predicted_labels[confident_mask].unsqueeze(1)
    data_pseudo_labeled = unsupervised_train_data[confident_mask]
    
    # Combine labeled and pseudo-labeled data
    data_train_combined = torch.concatenate([supervised_train_data, data_pseudo_labeled], axis=0)
    labels_train_combined = torch.concatenate([supervised_train_labels, pseudo_labels], axis=0)
    # ---------------

    # Train model again with both labeled and pseudo-labeled data
    model.fit(data_train_combined, labels_train_combined, epochs=10, validation_data=validation_set)

    # consistency training
    with tf.GradientTape() as tape:
        batch_before_becoming_y_pred = data_train_combined
        preds = model(batch_before_becoming_y_pred, training=True)  # Use training=True for gradient computation
        pred_probs = preds

        between_preds = []
        between_inputs = []
        for j in range(data_train_combined.shape[0]):  # Changed i to j to avoid conflict with outer loop
            if j == 0:
                continue

            between_preds.append((pred_probs[j] + pred_probs[j-1]) / 2)
            between_inputs.append((data_train_combined[j] + data_train_combined[j-1]) / 2)

        # Convert lists to tensors for batch processing
        between_inputs_tensor = tf.stack(between_inputs)
        between_preds_tensor = tf.stack(between_preds)
        
        # Get model predictions for interpolated inputs
        between_outputs = model(between_inputs_tensor, training=True)
        
        # Compute consistency loss as MSE between interpolated predictions and model outputs
        consistency_loss = tf.reduce_mean(tf.square(between_outputs - between_preds_tensor))

        
    gradients = tape.gradient(consistency_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    evaluation_test = model.evaluate(train_data_B, train_labels_B)
    evaluation_train = model.evaluate(supervised_train_data, supervised_train_labels)
    
    if WANDB_ENABLED:
        wandb.log({
            "Consistency loss": consistency_loss.numpy().item(),
            "test_loss": evaluation_test[0],
            "test_accuracy": evaluation_test[1],
            "train_loss": evaluation_train[0],
            "train_accuracy": evaluation_train[1],
            "percentage of pseudolabeled items": num_confident / len(unsupervised_train_data),
            "epoch": i
        })

if WANDB_ENABLED:
    run.finish()
