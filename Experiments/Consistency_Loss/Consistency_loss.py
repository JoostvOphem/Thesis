from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np
from transformers import RobertaModel, RobertaTokenizer

from data_utils import get_dataset

DATASET1 = "Ghostbusters_all"  # options: "gpt_writing", "monolingual_davinci", "GPT2", "Ghostbusters_all", "SemEval_complete"
DATASET2 = "SemEval_complete"
ROBERTA_USED = "Ghostbusters_all"

NORMALIZED = False 

### BEST PARAMETERS
# NORMALIZED = FALSE
# BATCH_SIZE = 64
# CONSISTENCY_MULTIPLIER = 1 # Set to 0 to disable consistency loss
# consistency_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.00001, weight_decay=0.01)
# fit (epochs=19) -> consistency_training_step

CONSISTENCY_MULTIPLIER = 1 # Set to 0 to disable consistency loss

WANDB_ENABLED = True
if WANDB_ENABLED:
    import wandb

    # run = wandb.init(
    #     project="CL_testing",
    #     name=f'30-05-{DATASET1}-{DATASET2}-{CONSISTENCY_MULTIPLIER}x_costum_loss'
    # )
    run = wandb.init(
        project="30-05-comparisons",
        name=f'CL_{DATASET1}_{DATASET2}'
    )

def consistency_training_step(model, 
                              data, 
                              consistency_weight, 
                              batch_size=32, 
                              epochs=10, 
                              shuffle=True):
    """
    Performs consistency training with batching similar to model.fit()
    
    Args:
        model: The TensorFlow model to train
        data: Training data tensor
        consistency_weight: Scalar weight for consistency loss
        batch_size: Batch size for training (default: 32)
        epochs: Number of epochs to train (default: 10)
        shuffle: Whether to shuffle data each epoch (default: True)
    
    Returns:
        List of consistency losses per epoch
    """
    
    consistency_losses = []
    num_samples = data.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle data at the beginning of each epoch
        if shuffle:
            indices = tf.random.shuffle(tf.range(num_samples))
            shuffled_data = tf.gather(data, indices)
        else:
            shuffled_data = data
        
        # Process data in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = shuffled_data[start_idx:end_idx]
            
            # Skip batches that are too small for interpolation
            if batch_data.shape[0] < 2:
                continue
                
            with tf.GradientTape() as tape:
                # Get predictions for current batch
                preds = model(batch_data, training=True)
                
                # Create interpolated inputs and predictions
                between_preds = []
                between_inputs = []
                
                for j in range(1, batch_data.shape[0]):  # Start from 1 to avoid index 0
                    # Interpolate between consecutive samples
                    between_preds.append((preds[j] + preds[j-1]) / 2)
                    between_inputs.append((batch_data[j] + batch_data[j-1]) / 2)
                
                # Skip if no interpolations possible
                if not between_inputs:
                    continue
                
                # Convert to tensors for batch processing
                between_inputs_tensor = tf.stack(between_inputs)
                between_preds_tensor = tf.stack(between_preds)
                
                # Get model predictions for interpolated inputs
                between_outputs = model(between_inputs_tensor, training=True)
                
                # Compute consistency loss
                batch_consistency_loss = consistency_weight * tf.reduce_mean(
                    tf.square(between_outputs - between_preds_tensor)
                )
            
            # Compute and apply gradients
            gradients = tape.gradient(batch_consistency_loss, model.trainable_variables)
            
            # Check for valid gradients
            if gradients and all(grad is not None for grad in gradients):
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                epoch_losses.append(batch_consistency_loss.numpy())
        
        # Average loss for this epoch
        if epoch_losses:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            consistency_losses.append(avg_epoch_loss)
            print(f"Consistency Training Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.6f}")
        else:
            consistency_losses.append(0.0)
            print(f"Consistency Training Epoch {epoch + 1}/{epochs} - No valid batches processed")
    
    return consistency_losses


def consistency_training_step_advanced(model, data, consistency_weight, optimizer, batch_size=32, epochs=10, 
                                     shuffle=True, interpolation_pairs='consecutive'):
    """
    Advanced consistency training with more interpolation options
    
    Args:
        model: The TensorFlow model to train
        data: Training data tensor
        consistency_weight: Scalar weight for consistency loss
        batch_size: Batch size for training
        epochs: Number of epochs to train
        shuffle: Whether to shuffle data each epoch
        interpolation_pairs: 'consecutive', 'random', or 'all_pairs'
    
    Returns:
        List of consistency losses per epoch
    """
    
    consistency_losses = []
    num_samples = data.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        epoch_losses = []
        
        if shuffle:
            indices = tf.random.shuffle(tf.range(num_samples))
            shuffled_data = tf.gather(data, indices)
        else:
            shuffled_data = data
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = shuffled_data[start_idx:end_idx]
            
            if batch_data.shape[0] < 2:
                continue
                
            with tf.GradientTape() as tape:
                preds = model(batch_data, training=True)
                
                between_preds = []
                between_inputs = []
                
                if interpolation_pairs == 'consecutive':
                    # Original approach: consecutive pairs
                    for j in range(1, batch_data.shape[0]):
                        between_preds.append((preds[j] + preds[j-1]) / 2)
                        between_inputs.append((batch_data[j] + batch_data[j-1]) / 2)
                        
                elif interpolation_pairs == 'random':
                    # Random pairs within batch
                    batch_size_actual = batch_data.shape[0]
                    num_pairs = min(batch_size_actual // 2, 10)  # Limit pairs to avoid memory issues
                    
                    for _ in range(num_pairs):
                        idx1, idx2 = tf.random.shuffle(tf.range(batch_size_actual))[:2]
                        between_preds.append((preds[idx1] + preds[idx2]) / 2)
                        between_inputs.append((batch_data[idx1] + batch_data[idx2]) / 2)
                
                if not between_inputs:
                    continue
                
                between_inputs_tensor = tf.stack(between_inputs)
                between_preds_tensor = tf.stack(between_preds)
                between_outputs = model(between_inputs_tensor, training=True)
                
                batch_consistency_loss = consistency_weight * tf.reduce_mean(
                    tf.square(between_outputs - between_preds_tensor)
                )
            
            gradients = tape.gradient(batch_consistency_loss, model.trainable_variables)
            
            if gradients and all(grad is not None for grad in gradients):
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                epoch_losses.append(batch_consistency_loss.numpy())
        
        if epoch_losses:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            consistency_losses.append(avg_epoch_loss)
            print(f"Consistency Training Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.6f}")
        else:
            consistency_losses.append(0.0)
            print(f"Consistency Training Epoch {epoch + 1}/{epochs} - No valid batches processed")
    
    return consistency_losses

def ss_model(loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=1e-3,
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

train_data_A, train_labels_A, val_data_A, val_labels_A, test_data_A, test_labels_A = get_dataset(DATASET1, ROBERTA_USED)

if DATASET1 == DATASET2: # use validation set of dataset A for dataset B if they are the same to prevent duplicate data.
    _, _, test_data_B, test_labels_B, train_data_B, train_labels_B = get_dataset(DATASET2, ROBERTA_USED)
else:
    train_data_B, train_labels_B, _, _, test_data_B, test_labels_B = get_dataset(DATASET2, ROBERTA_USED)


supervised_train_data = train_data_A[:int(supervised_percentage * len(train_data_A))]
supervised_train_labels = train_labels_A[:int(supervised_percentage * len(train_labels_A))]
unsupervised_train_data = train_data_B[int(supervised_percentage * len(train_data_B)):]
unsupervised_train_labels = train_labels_B[int(supervised_percentage * len(train_labels_B)):]

if NORMALIZED:
    supervised_train_data = torch.nn.functional.normalize(supervised_train_data)
    unsupervised_train_data = torch.nn.functional.normalize(unsupervised_train_data)
    val_data_A = torch.nn.functional.normalize(val_data_A)

validation_set = (val_data_A, val_labels_A)

consistency_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.00001, weight_decay=0.01)


model = ss_model()
for i in range(50):
    # ---------------
    # generate pseudo labels

    # Get prediction probabilities
    pred_probs = torch.tensor(model.predict(unsupervised_train_data))

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
    model.fit(data_train_combined, labels_train_combined, epochs=19, validation_data=validation_set)
    if True:
        if CONSISTENCY_MULTIPLIER == 0:
            consistency_loss = tf.constant(0.0)
        else:
            # consistency training
            consistency_losses = consistency_training_step_advanced(
                model,
                data_train_combined,
                CONSISTENCY_MULTIPLIER,
                batch_size=64,
                epochs=1,
                shuffle=True,
                interpolation_pairs='random',
                optimizer=consistency_optimizer
            )
            consistency_loss = tf.reduce_mean(consistency_losses, axis=0)
            consistency_loss = 1/CONSISTENCY_MULTIPLIER * consistency_loss.numpy().item()
    
    else:
        consistency_loss = tf.constant(0.0)
    # with tf.GradientTape() as tape:
    #     batch_before_becoming_y_pred = data_train_combined
    #     preds = model(batch_before_becoming_y_pred, training=True)  # Use training=True for gradient computation
    #     pred_probs = preds

    #     between_preds = []
    #     between_inputs = []
    #     for j in range(data_train_combined.shape[0]):  # Changed i to j to avoid conflict with outer loop
    #         if j == 0:
    #             continue

    #         between_preds.append((pred_probs[j] + pred_probs[j-1]) / 2)
    #         between_inputs.append((data_train_combined[j] + data_train_combined[j-1]) / 2)

    #     # Convert lists to tensors for batch processing
    #     between_inputs_tensor = tf.stack(between_inputs)
    #     between_preds_tensor = tf.stack(between_preds)
        
    #     # Get model predictions for interpolated inputs
    #     between_outputs = model(between_inputs_tensor, training=True)
        
    #     # Compute consistency loss as MSE between interpolated predictions and model outputs
    #     consistency_loss = CONSISTENCY_MULTIPLIER * tf.reduce_mean(tf.square(between_outputs - between_preds_tensor))

        
    # gradients = tape.gradient(consistency_loss, model.trainable_variables)
    # model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    test_evaluation_B = model.evaluate(test_data_B, test_labels_B)
    test_evaluation_A = model.evaluate(test_data_A, test_labels_A)

    train_evaluation_A = model.evaluate(train_data_A, train_labels_A)
    train_evaluation_B = model.evaluate(train_data_B, train_labels_B)
    
    # if WANDB_ENABLED:
    #     wandb.log({
    #         "Consistency loss": consistency_loss,
    #         "test_loss_B": test_evaluation_B[0],
    #         "test_accuracy_B": test_evaluation_B[1],
    #         "test_loss_A": test_evaluation_A[0],
    #         "test_accuracy_A": test_evaluation_A[1],
    #         "train_loss_A": train_evaluation_A[0],
    #         "train_accuracy_A": train_evaluation_A[1],
    #         "train_loss_B": train_evaluation_B[0],
    #         "train_accuracy_B": train_evaluation_B[1],
    #         "percentage of pseudolabeled items": num_confident / len(unsupervised_train_data),
    #         "epoch": i
    #     })
    if WANDB_ENABLED:
        wandb.log({
            "train_loss": train_evaluation_A[0],
            "train_accuracy": train_evaluation_A[1],
            "test_loss": test_evaluation_B[0],
            "test_accuracy": test_evaluation_B[1],
            "percentage of pseudolabeled items": num_confident / len(unsupervised_train_data),
            "epoch": i
        })

if WANDB_ENABLED:
    run.finish()
