from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np

from data_utils import get_dataset

DATASET1 = "Ghostbusters_all"  # options: "Ghostbusters_all", "gpt_writing", "monolingual_davinci", "GPT2", "SemEval_complete"
DATASET2 = "monolingual_complete"
DATASET3 = "gpt2"
ROBERTA_USED = "Ghostbusters_all"

for i in range(10, 20):

    WANDB_ENABLED = True
    if WANDB_ENABLED:
        import wandb

        run = wandb.init(
            project="Transfer-Learning",
            name=f'SSL A={DATASET1}, B={DATASET2}, C={DATASET3}_{i+1}',
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

    train_data_A, train_labels_A, val_data_A, val_labels_A, test_data_A, test_labels_A = get_dataset(DATASET1, ROBERTA_USED)

    if DATASET1 == DATASET2: # use validation set of dataset A for dataset B if they are the same to prevent duplicate data.
        _, _, _, _, train_data_B, train_labels_B = get_dataset(DATASET2, ROBERTA_USED)
    else:
        train_data_B, train_labels_B, _, _, test_data_B, test_labels_B = get_dataset(DATASET2, ROBERTA_USED)

    _, _, _, _, test_data_C, test_labels_C = get_dataset(DATASET3, ROBERTA_USED)

    # train_data_A = torch.nn.functional.normalize(train_data_A)
    # val_data_A = torch.nn.functional.normalize(val_data_A)
    # train_data_B = torch.nn.functional.normalize(train_data_B)

    supervised_train_data = train_data_A[:int(supervised_percentage * len(train_data_A))]
    supervised_train_labels = train_labels_A[:int(supervised_percentage * len(train_labels_A))]
    unsupervised_train_data = train_data_B[int(supervised_percentage * len(train_data_B)):]
    unsupervised_train_labels = train_labels_B[int(supervised_percentage * len(train_labels_B)):]

    validation_set = (val_data_A, val_labels_A)

    model = ss_model()

    for i in range(20):
        # ---------------
        # generate pseudo labels

        # Get prediction probabilities
        pred_probs = torch.tensor(model.predict(unsupervised_train_data))  # shape: (N, num_classes)

        # Get max probabilities and predicted labels
        max_probs, predicted_labels = torch.max(pred_probs, dim=1)

        # Filter for high-confidence predictions (>= 0.99)

        entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-8), dim=1)
        max_entropy = torch.log(torch.tensor(2.0))  # For binary classification
        confidence = 1 - (entropy / max_entropy)
        confident_mask = confidence >= 0.99

        # confident_mask = max_probs >= 0.99
        num_confident = confident_mask.sum().item()
        pseudo_labels = predicted_labels[confident_mask].unsqueeze(1)
        data_pseudo_labeled = unsupervised_train_data[confident_mask]
        
        # Combine labeled and pseudo-labeled data
        data_train_combined = torch.concatenate([supervised_train_data, data_pseudo_labeled], axis=0)
        labels_train_combined = torch.concatenate([supervised_train_labels, pseudo_labels], axis=0)
        # ---------------

        # Train model again with both labeled and pseudo-labeled data
        model.fit(data_train_combined, labels_train_combined, epochs=20, validation_data=validation_set)

        evaluation_test_A = model.evaluate(test_data_A, test_labels_A)
        evaluation_train_A = model.evaluate(train_data_A, train_labels_A)
        
        evaluation_test_B = model.evaluate(test_data_B, test_labels_B)
        evaluation_train_B = model.evaluate(supervised_train_data, supervised_train_labels)

        evaluation_test_C = model.evaluate(test_data_C, test_labels_C)

        if WANDB_ENABLED:
            wandb.log({
                "test_loss_A": evaluation_test_A[0],
                "test_accuracy_A": evaluation_test_A[1],
                "train_loss_A": evaluation_train_A[0],
                "train_accuracy_A": evaluation_train_A[1],
                "test_loss_B": evaluation_test_B[0],
                "test_accuracy_B": evaluation_test_B[1],
                "train_loss_B": evaluation_train_B[0],
                "train_accuracy_B": evaluation_train_B[1],
                "test_loss_C": evaluation_test_C[0],
                "test_accuracy_C": evaluation_test_C[1],
                "percentage of pseudolabeled items": num_confident / len(unsupervised_train_data),
                "epoch": i
            })

    run.finish()