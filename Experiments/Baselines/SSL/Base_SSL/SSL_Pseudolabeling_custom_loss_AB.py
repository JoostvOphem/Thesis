from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np
from transformers import RobertaModel, RobertaTokenizer


DATASET1 = "GPT2"  # Current options: "gpt_writing", "GPT2", "monolingual_davinci"
DATASET2 = "monolingual_davinci"

# model_name = 'roberta-base'
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# roberta = RobertaModel.from_pretrained(model_name)

# # classifier = Classifier(hidden_dim=768)

# state_dict = torch.load(f"best_roberta_monolingual_300/best_classifier.pt")
# roberta.load_state_dict(state_dict)

WANDB_ENABLED = False
if WANDB_ENABLED:
    import wandb

    run = wandb.init(
        project="CL_testing",
        name=f'27-05-{DATASET1}-{DATASET2}-joined_costum_loss'
    )

def custom_sparse_categorical_crossentropy(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

def consistency_loss(y_true, y_pred):
    """
    Computes the consistency loss for a batch of predictions y_pred.
    For each pair (A, B) in the batch, compares the average of predictions to the prediction of the average embedding.
    """
    # between_preds = []
    # between_inputs = []
    # for i in range(y_pred.shape[0]):
    #     if i == 0:
    #         continue

    #     between_preds.append((y_pred[i] + y_pred[i-1]) / 2)
    #     between_inputs.append((batch_before_becoming_y_pred[i] + batch_before_becoming_y_pred[i+1]) / 2)
    
    # loss = []
    
    # for between_input, between_pred in zip(between_inputs, between_preds):
    #     between_output = MODEL.foward(between_input)
    #     loss.append(difference_between(between_output, between_pred))
    
    # return loss
    return custom_sparse_categorical_crossentropy(y_true, y_pred)

class InnerLossTracker():
    def __init__(self):
        self.loss1_amts = []
        self.loss2_amts = []

INNER_LOSS_TRACKER = InnerLossTracker()

def difference_between(preds, C):
    """
    Computes the difference between the predictions and a constant C.
    This is used to track the consistency loss.
    """
    a, b = preds
    mid = (a + b) / 2
    return mid - C

def append_loss(loss, y_true, y_pred):
    """
    This function is used to append a test value to the loss2_amts list in the INNER_LOSS_TRACKER.
    It is called within the custom_loss function to track the consistency loss.
    """
    INNER_LOSS_TRACKER.loss2_amts.append((loss.shape, y_true.shape, y_pred))
    return 0

def custom_loss(y_true, y_pred, cl_loss_percent=0.1):
    sparse_caterogical_crossentropy_loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred)
    consistency_amt = consistency_loss(y_true, y_pred)

    tf.py_function(
        append_loss,
        [consistency_amt, y_true, y_pred],
        tf.int32
        )

    return ((1-cl_loss_percent) * sparse_caterogical_crossentropy_loss + 
             cl_loss_percent * consistency_amt)

def ss_model(loss_function = custom_loss,
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

print("model being created")
model = ss_model()
print("in fore loop")
for i in range(1):
    print(f"Epoch {i+1} of 50")
    # ---------------
    # generate pseudo labels

    # Get prediction probabilities
    pred_probs = torch.tensor(model.predict(unsupervised_train_data))  # shape: (N, num_classes)

    print("probs_predded")

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
    print("data combined")
    # ---------------

    # Train model again with both labeled and pseudo-labeled data
    model.fit(data_train_combined, labels_train_combined, epochs=10, validation_data=validation_set)

    # consistency training
    with tf.GradientTape() as tape:
        batch_before_becoming_y_pred = data_train_combined
        preds = model.predict(batch_before_becoming_y_pred)
        pred_probs = preds

        between_preds = []
        between_inputs = []
        for i in range(data_train_combined.shape[0]):
            if i == 0:
                continue

            between_preds.append((pred_probs[i] + pred_probs[i-1]) / 2)
            between_inputs.append((data_train_combined[i] + data_train_combined[i-1]) / 2)

        losses = []
        for between_input, between_pred in zip(between_inputs, between_preds):
            between_output = model(between_input)
            loss = (between_output - between_pred) / 2
            losses.append(loss)
        
    gradients = tape.gradient(losses, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print("\n\nmodel fitted\n")

    evaluation_test = model.evaluate(train_data_B, train_labels_B)
    evaluation_train = model.evaluate(supervised_train_data, supervised_train_labels)

    print(type(tf.reduce_mean(loss, axis=0, keepdims=True)))
    
    if WANDB_ENABLED:
        wandb.log({
            "Consistency loss": tf.reduce_mean(loss, axis=0, keepdims=True),
            "test_loss": evaluation_test[0],
            "test_accuracy": evaluation_test[1],
            "train_loss": evaluation_train[0],
            "train_accuracy": evaluation_train[1],
            "percentage of pseudolabeled items": num_confident / len(unsupervised_train_data),
            "epoch": i
        })

if WANDB_ENABLED:
    run.finish()
