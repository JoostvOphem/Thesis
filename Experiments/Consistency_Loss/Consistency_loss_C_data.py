from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np
import math
from transformers import RobertaModel, RobertaTokenizer

from data_utils import get_dataset, get_jsonl_paths_of_dataset, get_roberta_and_tokenizer, embed_layer, join_sentences

DATASET1 = "claude"  # options: "gpt_writing", "monolingual_davinci", "GPT2", "Ghostbusters_all", "SemEval_complete"
DATASET2 = "gpt_prompt2"
DATASET3 = "gpt2"
ROBERTA_USED = "claude"

        # A: Claude
        # B: gpt_prompt2
        # C: gpt2

MERGE_METHOD = 'word'
NORMALIZED = True 

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

    run = wandb.init(
        project=f"different_consistency_types",
        name=f'CL A={DATASET1}, B={DATASET2}, C={DATASET3}, way={MERGE_METHOD}',
    )

def consistency_training_step_advanced_unembedded(
        model,
        roberta,
        tokenizer,
        text_train_data_combined, 
        consistency_weight, 
        optimizer,
        embed_function, 
        paragraph_merge_function,
        merge_method,
        batch_size=32, 
        epochs=10,
        shuffle=True, 
        interpolation_pairs='consecutive'):
    """
    Advanced consistency training with text-based interpolation
    Args:
        model: The TensorFlow model to train (head model)
        text_train_data_combined: List of strings (text data)
        consistency_weight: Scalar weight for consistency loss
        optimizer: Optimizer for training
        embed_function: Function that converts text to embeddings
        paragraph_merge_function: Function that merges two paragraphs
        batch_size: Batch size for training
        epochs: Number of epochs to train
        shuffle: Whether to shuffle data each epoch
        interpolation_pairs: 'consecutive', 'random', or 'all_pairs'
    Returns:
        List of consistency losses per epoch
    """
    consistency_losses = []
    num_samples = len(text_train_data_combined)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle text data if requested
        if shuffle:
            import random
            shuffled_texts = text_train_data_combined.copy()
            random.shuffle(shuffled_texts)
        else:
            shuffled_texts = text_train_data_combined
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_texts = shuffled_texts[start_idx:end_idx]
            
            if len(batch_texts) < 2:
                continue
            
            # Embed the batch texts
            print(f"epoch: {epoch+1} / {epochs}, batch: {batch_idx+1} / {num_batches}")
            batch_embeddings = tf.stack([embed_function(roberta, tokenizer, text) for text in batch_texts])
            
            with tf.GradientTape() as tape:
                # Get predictions for original texts
                preds = model(batch_embeddings, training=True)
                
                # Generate merged texts and their predictions
                merged_texts = []
                between_preds = []
                
                if interpolation_pairs == 'consecutive':
                    # Consecutive pairs
                    for j in range(1, len(batch_texts)):
                        merged_text = paragraph_merge_function(batch_texts[j-1], batch_texts[j], way=merge_method)
                        merged_texts.append(merged_text)
                        between_preds.append((preds[j-1] + preds[j]) / 2)
                        
                elif interpolation_pairs == 'random':
                    # Random pairs within batch
                    batch_size_actual = len(batch_texts)
                    num_pairs = min(batch_size_actual // 2, 10)  # Limit pairs to avoid memory issues
                    
                    for _ in range(num_pairs):
                        # Get two random indices
                        indices = tf.random.shuffle(tf.range(batch_size_actual))[:2]
                        idx1, idx2 = indices[0].numpy(), indices[1].numpy()
                        
                        merged_text = paragraph_merge_function(batch_texts[idx1], batch_texts[idx2], way=merge_method)
                        merged_texts.append(merged_text)
                        between_preds.append((preds[idx1] + preds[idx2]) / 2)
                
                if not merged_texts:
                    continue
                
                # Embed merged texts and get their predictions
                print(f"embedding {len(merged_texts)} merged texts")
                merged_embeddings = tf.stack([embed_function(roberta, tokenizer, text) for text in merged_texts])
                merged_outputs = model(merged_embeddings, training=True)
                
                # Stack the interpolated predictions
                between_preds_tensor = tf.stack(between_preds)
                
                # Calculate consistency loss
                batch_consistency_loss = consistency_weight * tf.reduce_mean(
                    tf.square(merged_outputs - between_preds_tensor)
                )
                
                # Apply gradients
                gradients = tape.gradient(batch_consistency_loss, model.trainable_variables)
                if gradients and all(grad is not None for grad in gradients):
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                epoch_losses.append(batch_consistency_loss.numpy())
        
        # Calculate and log epoch loss
        if epoch_losses:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            consistency_losses.append(avg_epoch_loss)
            print(f"Consistency Training Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.6f}")
        else:
            consistency_losses.append(0.0)
            print(f"Consistency Training Epoch {epoch + 1}/{epochs} - No valid batches processed")
    
    return consistency_losses

# def consistency_training_step_advanced_unembedded(model, roberta, tokenizer, data, consistency_weight, optimizer, batch_size=32, epochs=10, 
#                                     shuffle=True, interpolation_pairs='consecutive'):
#     """
#     Advanced consistency training with more interpolation options
    
#     Args:
#         model: The TensorFlow model to train
#         data: Training data tensor
#         consistency_weight: Scalar weight for consistency loss
#         batch_size: Batch size for training
#         epochs: Number of epochs to train
#         shuffle: Whether to shuffle data each epoch
#         interpolation_pairs: 'consecutive', 'random', or 'all_pairs'
    
#     Returns:
#         List of consistency losses per epoch
#     """
    
#     consistency_losses = []
#     num_samples = data.shape[0]
#     num_batches = (num_samples + batch_size - 1) // batch_size
    
#     for epoch in range(epochs):
#         epoch_losses = []
        
#         if shuffle:
#             indices = tf.random.shuffle(tf.range(num_samples))
#             shuffled_data = tf.gather(data, indices)
#         else:
#             shuffled_data = data
        
#         for batch_idx in range(num_batches):
#             start_idx = batch_idx * batch_size
#             end_idx = min(start_idx + batch_size, num_samples)
#             batch_data = shuffled_data[start_idx:end_idx]
            
#             if batch_data.shape[0] < 2:
#                 continue
                
#             with tf.GradientTape() as tape:
#                 preds = model(batch_data, training=True)
                
#                 between_preds = []
#                 between_inputs = []
                
#                 if interpolation_pairs == 'consecutive':
#                     # Original approach: consecutive pairs
#                     for j in range(1, batch_data.shape[0]):
#                         between_preds.append((preds[j] + preds[j-1]) / 2)
#                         between_inputs.append((batch_data[j] + batch_data[j-1]) / 2)
                        
#                 elif interpolation_pairs == 'random':
#                     # Random pairs within batch
#                     batch_size_actual = batch_data.shape[0]
#                     num_pairs = min(batch_size_actual // 2, 10)  # Limit pairs to avoid memory issues
                    
#                     for _ in range(num_pairs):
#                         idx1, idx2 = tf.random.shuffle(tf.range(batch_size_actual))[:2]
#                         between_preds.append((preds[idx1] + preds[idx2]) / 2)
#                         between_inputs.append((batch_data[idx1] + batch_data[idx2]) / 2)
                
#                 if not between_inputs:
#                     continue
                
#                 between_inputs_tensor = tf.stack(between_inputs)
#                 between_preds_tensor = tf.stack(between_preds)
#                 between_outputs = model(between_inputs_tensor, training=True)
                
#                 batch_consistency_loss = consistency_weight * tf.reduce_mean(
#                     tf.square(between_outputs - between_preds_tensor)
#                 )
            
#             gradients = tape.gradient(batch_consistency_loss, model.trainable_variables)
            
#             if gradients and all(grad is not None for grad in gradients):
#                 optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#                 epoch_losses.append(batch_consistency_loss.numpy())
        
#         if epoch_losses:
#             avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
#             consistency_losses.append(avg_epoch_loss)
#             print(f"Consistency Training Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.6f}")
#         else:
#             consistency_losses.append(0.0)
#             print(f"Consistency Training Epoch {epoch + 1}/{epochs} - No valid batches processed")
    
#     return consistency_losses

class ConsistencyLRScheduler:
    def __init__(self, initial_lr=1e-3, schedule_type='cosine', total_epochs=20):
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
    
    def get_lr(self, epoch):
        if self.schedule_type == 'cosine':
            return self.initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))
        elif self.schedule_type == 'step':
            if epoch < 8: return self.initial_lr
            elif epoch < 16: return self.initial_lr * 0.1
            else: return self.initial_lr * 0.01

def apply_mask(lst, msk):
    out = []
    for i, bool in enumerate(msk):
        if bool:
            out.append(lst[i])
    
    return out

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
    _, _, test_data_B_embedded, test_labels_B_embedded, train_data_B_embedded, train_labels_B_embedded = get_dataset(DATASET2, ROBERTA_USED)
else:
    train_data_B_embedded, train_labels_B_embedded, _, _, test_data_B_embedded, test_labels_B_embedded = get_dataset(DATASET2, ROBERTA_USED)

def get_train_texts_labels(dataset, amt):
    jsonl_train, _, _ = get_jsonl_paths_of_dataset(dataset)
    train_jsonOBJ = pd.read_json(jsonl_train, lines=True).reset_index(drop=True)

    labels = []
    texts = []

    # not shuffled in the embedding file, so these texts correspond to the embeddings
    # allowing us to make a mask on the embeddings and using it on these texts

    for row in train_jsonOBJ.iterrows():
        label = int(row[1]['label'])
        text = row[1]['text']

        labels.append(label)
        texts.append(text)
    
    return texts[:amt], labels[:amt]

texts_A, labels_A = get_train_texts_labels(DATASET1, len(train_labels_A))
texts_B, labels_B = get_train_texts_labels(DATASET2, len(train_labels_B_embedded))

roberta, tokenizer = get_roberta_and_tokenizer(DATASET1)


_, _, _, _, test_data_C, test_labels_C = get_dataset(DATASET3, ROBERTA_USED)

supervised_train_data = train_data_A[:int(supervised_percentage * len(train_data_A))]
supervised_train_labels = train_labels_A[:int(supervised_percentage * len(train_labels_A))]
unsupervised_train_data = train_data_B_embedded[int(supervised_percentage * len(train_data_B_embedded)):]
unsupervised_train_labels = train_labels_B_embedded[int(supervised_percentage * len(train_labels_B_embedded)):]

if NORMALIZED:
    supervised_train_data = torch.nn.functional.normalize(supervised_train_data)
    unsupervised_train_data = torch.nn.functional.normalize(unsupervised_train_data)
    val_data_A = torch.nn.functional.normalize(val_data_A)

validation_set = (val_data_A, val_labels_A)

# consistency_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.01) # was: lr=0.00001, wd=0.01
lr_scheduler = ConsistencyLRScheduler(initial_lr=1e-2, schedule_type='cosine', total_epochs=20)

model = ss_model()
for i in range(20):
    # ---------------
    # generate pseudo labels

    # Get prediction probabilities
    print(type(unsupervised_train_data), unsupervised_train_data.shape)
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

    B_mask = confident_mask.tolist()
    texts_B_pseudo_labeled = apply_mask(texts_B, B_mask)
    labels_B_pseudo_labeled = apply_mask(labels_B, B_mask)


    text_train_data_combined = texts_A + texts_B_pseudo_labeled
    
    # Combine labeled and pseudo-labeled data
    data_train_combined = torch.concatenate([supervised_train_data, data_pseudo_labeled], axis=0)
    labels_train_combined = torch.concatenate([supervised_train_labels, pseudo_labels], axis=0)
    # ---------------

    if CONSISTENCY_MULTIPLIER == 0:
        consistency_loss = tf.constant(0.0)
    else:
        current_lr = lr_scheduler.get_lr(i)
        consistency_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=current_lr,
            weight_decay=0.01) # was: lr=0.00001, wd=0.01
        # consistency training

        consistency_losses = consistency_training_step_advanced_unembedded(
            model,
            roberta,
            tokenizer,
            text_train_data_combined,
            CONSISTENCY_MULTIPLIER,
            optimizer=consistency_optimizer,
            embed_function=embed_layer,
            paragraph_merge_function=join_sentences,
            merge_method=MERGE_METHOD,
            batch_size=16,
            epochs=1,
            shuffle=True,
            interpolation_pairs='random',
        )
        consistency_loss = tf.reduce_mean(consistency_losses, axis=0)
        consistency_loss = 1/CONSISTENCY_MULTIPLIER * consistency_loss.numpy().item()

    # Train model again with both labeled and pseudo-labeled data
    model.fit(data_train_combined, labels_train_combined, epochs=19, validation_data=validation_set)

    evaluation_test_A = model.evaluate(test_data_A, test_labels_A)
    evaluation_train_A = model.evaluate(train_data_A, train_labels_A)
    
    evaluation_test_B = model.evaluate(test_data_B_embedded, test_labels_B_embedded)
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
            "epoch": i,
            "consistency_loss": consistency_loss
        })

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
    # if WANDB_ENABLED:
    #     wandb.log({
    #         "consistency_loss": consistency_loss,
    #         "train_loss": train_evaluation_A[0],
    #         "train_accuracy": train_evaluation_A[1],
    #         "test_loss": test_evaluation_B[0],
    #         "test_accuracy": test_evaluation_B[1],
    #         "test_loss_C": test_evaluation_C[0],
    #         "test_accuracy_C": test_evaluation_C[1],
    #         "percentage of pseudolabeled items": num_confident / len(unsupervised_train_data),
    #         "epoch": i
    #     })

if WANDB_ENABLED:
    run.finish()
