from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np



WANDB_ENABLED = False
if WANDB_ENABLED:
    import wandb

    run = wandb.init(
        project="SSL_Roberta_compare_AB",
        name='SemEval_datasetB_promising'
    )

def ss_model(loss_function = "sparse_categorical_crossentropy",
             optimizer = "adamw",
             metrics=["accuracy"],
             epochs=30,
             layers = [tf.keras.layers.Flatten(input_shape=[768,1], name="inputLayer"),
                       tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
                       tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
                       tf.keras.layers.Dense(10, activation="relu", name="hiddenlayer3"),
                       tf.keras.layers.Dense(2, activation="softmax", name="outputlayer")]
            ):

    model = tf.keras.models.Sequential(layers)
    
    model.compile(loss=loss_function,
                 optimizer=optimizer,
                 metrics=metrics)

    return model

supervised_amt = 0.1
import json

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


supervised_train_data_A = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_train.jsonl")))
supervised_train_labels_A = label_data("Datasets/Ghostbusters_standardized/gpt_writing_train.jsonl")

val_data_A = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_val.jsonl")))
val_labels_A = label_data("Datasets/Ghostbusters_standardized/gpt_writing_val.jsonl")

test_data_A = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_test.jsonl")))
test_labels_A = label_data("Datasets/Ghostbusters_standardized/gpt_writing_test.jsonl")


# shuffle the data
def shuffle_data(data, labels):
    indices = torch.randperm(len(data))
    return data[indices], labels[indices]

supervised_train_data_A, supervised_train_labels_A = shuffle_data(supervised_train_data_A, supervised_train_labels_A)
val_data_A, val_labels_A = shuffle_data(val_data_A, val_labels_A)
test_data_A, test_labels_A = shuffle_data(test_data_A, test_labels_A)

validation_set_A = (val_data_A, val_labels_A)

supervised_train_data_B = torch.tensor(np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_train.jsonl")))
supervised_train_labels_B = label_data("Datasets/SemEval_standardized/monolingual/monolingual_davinci_train.jsonl")


val_data_B = torch.tensor(np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_val.jsonl")))
val_labels_B = label_data("Datasets/SemEval_standardized/monolingual/monolingual_davinci_val.jsonl")

test_data_B = torch.tensor(np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_test.jsonl")))
test_labels_B = label_data("Datasets/SemEval_standardized/monolingual/monolingual_davinci_test.jsonl")

# shuffle the data
def shuffle_data(data, labels):
    indices = torch.randperm(len(data))
    return data[indices], labels[indices]

unsupervised_train_data_B, unsupervised_train_labels_B = shuffle_data(supervised_train_data_B, supervised_train_labels_B)
val_data_B, val_labels_B = shuffle_data(val_data_B, val_labels_B)
test_data_B, test_labels_B = shuffle_data(test_data_B, test_labels_B)

validation_set_B = (val_data_B, val_labels_B)


model = ss_model()


print("Original data shapes:")
print(f"Train data: {supervised_train_data_A.shape}")
print(f"Train labels: {supervised_train_labels_A.shape}")

# Fix data shapes: [1400, 1, 768] -> [1400, 768] and convert to numpy
train_data_A = supervised_train_data_A.squeeze(1).numpy()
train_labels_A = supervised_train_labels_A.numpy().astype(np.int32)  # Ensure integer labels

val_data_A_np = val_data_A.squeeze(1).numpy()
val_labels_A_np = val_labels_A.numpy().astype(np.int32)

print("Fixed data shapes:")
print(f"Train data: {train_data_A.shape}")
print(f"Train labels: {train_labels_A.shape}")
print(f"Val data: {val_data_A_np.shape}")
print(f"Val labels: {val_labels_A_np.shape}")

# Check data properties
print("Data diagnostics:")
print(f"Train data min/max: {train_data_A.min():.4f}, {train_data_A.max():.4f}")
print(f"Train data mean/std: {train_data_A.mean():.4f}, {train_data_A.std():.4f}")
print(f"Label distribution: {np.bincount(train_labels_A)}")
print(f"Unique labels: {np.unique(train_labels_A)}")

# Create model with correct architecture
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(300, activation="relu", input_shape=(768,), name="hiddenLayer1"),
        tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
        tf.keras.layers.Dense(10, activation="relu", name="hiddenLayer3"),
        tf.keras.layers.Dense(2, activation="softmax", name="outputLayer")
    ])
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model

# Create fresh model
model = create_model()
model.summary()

# Prepare validation data
validation_data = (val_data_A_np, val_labels_A_np)

print("\n=== STARTING TRAINING ===")
print("Step 1: Training on labeled Dataset A...")

# Train the model
history = model.fit(
    train_data_A, 
    train_labels_A,
    epochs=30, 
    validation_data=validation_data,
    batch_size=32,
    verbose=1
)

print("Training completed!")

# Test on a small batch to verify the model is working
print("\n=== TESTING MODEL ===")
test_predictions = model.predict(train_data_A[:10])
print(f"Sample predictions shape: {test_predictions.shape}")
print(f"Sample predictions: {test_predictions[:3]}")
print(f"Sample true labels: {train_labels_A[:3]}")

# # Initial evaluation on Dataset B
# initial_eval = model.evaluate(test_data_B, test_labels_B)
# print(f"Initial performance on Dataset B - Loss: {initial_eval[0]:.4f}, Accuracy: {initial_eval[1]:.4f}")

# # STEP 2: Semi-supervised iterations with pseudo-labeling on Dataset B
# for i in range(50):
#     print(f"\n--- Iteration {i+1} ---")
    
#     # Generate pseudo-labels on unlabeled Dataset B
#     pred_probs = torch.tensor(model.predict(unsupervised_train_data_B))
#     max_probs, predicted_labels = torch.max(pred_probs, dim=1)
    
#     # Use adaptive confidence threshold - start conservative due to domain shift
#     confidence_threshold = max(0.6, 0.8 - i * 0.002)  # Start at 0.8, decrease to 0.6
#     confident_mask = max_probs >= confidence_threshold
    
#     if confident_mask.sum() == 0:
#         print(f"No confident predictions with threshold {confidence_threshold:.3f}")
#         print("Continuing to next iteration...")
#         continue
    
#     pseudo_labels = predicted_labels[confident_mask]
#     data_pseudo_labeled = unsupervised_train_data_B[confident_mask]
    
#     print(f"Pseudo-labeled samples: {len(pseudo_labels)}/{len(unsupervised_train_data_B)} (threshold: {confidence_threshold:.3f})")
    
#     # Combine original labeled Dataset A with pseudo-labeled Dataset B
#     data_train_combined = torch.concatenate([supervised_train_data_A, data_pseudo_labeled], axis=0)
#     labels_train_combined = torch.concatenate([supervised_train_labels_A, pseudo_labels], axis=0)
    
#     print(f"Training on {len(supervised_train_data_A)} labeled + {len(data_pseudo_labeled)} pseudo-labeled samples")
    
#     # Train with combined data - use Dataset B validation
#     model.fit(data_train_combined, labels_train_combined, 
#               epochs=5, validation_data=validation_set_B, verbose=0)
    
#     # Evaluate on Dataset B test set
#     evaluation = model.evaluate(test_data_B, test_labels_B, verbose=0)
#     print(f"Performance on Dataset B - Loss: {evaluation[0]:.4f}, Accuracy: {evaluation[1]:.4f}")
    
#     # Optional: Check pseudo-label quality (for analysis only)
#     if len(pseudo_labels) > 0:
#         # Get true labels for the pseudo-labeled samples (just for monitoring)
#         true_labels_subset = supervised_train_labels_B[confident_mask]
#         pseudo_accuracy = (pseudo_labels == true_labels_subset).float().mean()
#         print(f"Pseudo-label accuracy: {pseudo_accuracy:.4f} (for monitoring only)")
    
#     if WANDB_ENABLED:
#         wandb.log({
#             "loss": evaluation[0],
#             "accuracy": evaluation[1],
#             "pseudo_samples": len(pseudo_labels),
#             "confidence_threshold": confidence_threshold,
#             "pseudo_accuracy": pseudo_accuracy.item() if len(pseudo_labels) > 0 else 0,
#             "iteration": i
#         })

# print("\nFinal evaluation:")
# final_eval = model.evaluate(test_data_B, test_labels_B)
# print(f"Final performance on Dataset B - Loss: {final_eval[0]:.4f}, Accuracy: {final_eval[1]:.4f}")
# print(f"Improvement: {final_eval[1] - initial_eval[1]:.4f}")

# # for i in range(50):
# #     # ---------------
# #     # generate pseudo labels

# #     # Get prediction probabilities
# #     pred_probs = torch.tensor(model.predict(unsupervised_train_data_B))  # shape: (N, num_classes)

# #     # Get max probabilities and predicted labels
# #     max_probs, predicted_labels = torch.max(pred_probs, dim=1)

# #     # Filter for high-confidence predictions (>= 0.95)
# #     confident_mask = max_probs >= 0.95
# #     pseudo_labels = predicted_labels[confident_mask]
# #     data_pseudo_labeled = unsupervised_train_data_B[confident_mask]
    
# #     # Combine labeled and pseudo-labeled data
# #     data_train_combined = torch.concatenate([supervised_train_data_A, data_pseudo_labeled], axis=0)
# #     labels_train_combined = torch.concatenate([supervised_train_labels_A, pseudo_labels], axis=0)
# #     # ---------------

# #     # Train model again with both labeled and pseudo-labeled data
# #     model.fit(data_train_combined, labels_train_combined, epochs=10, validation_data=validation_set_A)

# #     evaluation = model.evaluate(test_data_B, test_labels_B)
# #     print(model.metrics_names)
# #     print(evaluation)

# #     if WANDB_ENABLED:
# #         wandb.log({
# #             "loss": evaluation[0],
# #             "accuracy": evaluation[1],
# #             "epoch": i
# #         })