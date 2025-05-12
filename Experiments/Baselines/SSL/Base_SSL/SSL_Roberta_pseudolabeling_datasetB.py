from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np



WANDB_ENABLED = True
if WANDB_ENABLED:
    import wandb

    run = wandb.init(
        project="SSL_Roberta_pseudolabeling",
        name='SemEval_datasetB_more_data'
    )

# data = torch.tensor(np.array(torch.load("subset_embeddings.npy")))
# labels = torch.load("subset_labels.npy")

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


# data = torch.tensor(np.array(torch.load("subset_embeddings.npy")))
# labels = torch.load("subset_labels.npy")

# train, test, val = 0.6, 0.3, 0.1 # percentages
supervised_amt = 0.1

# train_size = int(train * len(data))
# val_size = int(val * len(data))

# dataset_A
train_data_human_A = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/human_embeded_subtaskA_train_monolingual.jsonl")))
test_data_human_A = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/human_embeded_subtaskA_test_monolingual.jsonl")))
val_data_human_A =torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/human_embeded_subtaskA_dev_monolingual.jsonl")))
train_labels_human_A = torch.zeros(len(train_data_human_A))
test_labels_human_A = torch.zeros(len(test_data_human_A))
val_labels_human_A = torch.zeros(len(val_data_human_A))

train_data_ai_A = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/ai_embeded_subtaskA_train_monolingual.jsonl")))
test_data_ai_A = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/ai_embeded_subtaskA_test_monolingual.jsonl")))
val_data_ai_A = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/ai_embeded_subtaskA_dev_monolingual.jsonl")))
train_labels_ai_A = torch.ones(len(train_data_ai_A))
test_labels_ai_A = torch.ones(len(test_data_ai_A))
val_labels_ai_A = torch.ones(len(val_data_ai_A))
supervised_train_data_human_A = train_data_human_A[:int(supervised_amt * len(train_data_human_A))]
supervised_train_labels_human_A = train_labels_human_A[:int(supervised_amt * len(train_labels_human_A))]
supervised_train_data_ai_A = train_data_ai_A[:int(supervised_amt * len(train_data_ai_A))]
supervised_train_labels_ai_A = train_labels_ai_A[:int(supervised_amt * len(train_labels_ai_A))]

supervised_train_data_A = torch.cat((supervised_train_data_human_A, supervised_train_data_ai_A), dim=0)
supervised_train_labels_A = torch.cat((supervised_train_labels_human_A, supervised_train_labels_ai_A), dim=0)

val_data_A = torch.cat((val_data_human_A, val_data_ai_A), dim=0)
val_labels_A = torch.cat((val_labels_human_A, val_labels_ai_A), dim=0)

test_data_A = torch.cat((test_data_human_A, test_data_ai_A), dim=0)
test_labels_A = torch.cat((test_labels_human_A, test_labels_ai_A), dim=0)

# shuffle the data
def shuffle_data(data, labels):
    indices = torch.randperm(len(data))
    return data[indices], labels[indices]

supervised_train_data_A, supervised_train_labels_A = shuffle_data(supervised_train_data_A, supervised_train_labels_A)
# unsupervised_train_data_A, unsupervised_train_labels_A = shuffle_data(unsupervised_train_data_A, unsupervised_train_labels_A)
val_data_A, val_labels_A = shuffle_data(val_data_A, val_labels_A)
test_data_A, test_labels_A = shuffle_data(test_data_A, test_labels_A)

validation_set_A = (val_data_A, val_labels_A)


# Dataset B

train_data_human_B = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/human_embeded_subtaskA_train_multilingual.jsonl")))
test_data_human_B = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/human_embeded_subtaskA_test_multilingual.jsonl")))
val_data_human_B =torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/human_embeded_subtaskA_dev_multilingual.jsonl")))
train_labels_human_B = torch.zeros(len(train_data_human_B))
test_labels_human_B = torch.zeros(len(test_data_human_B))
val_labels_human_B = torch.zeros(len(val_data_human_B))
train_data_ai_B = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/ai_embeded_subtaskA_train_multilingual.jsonl")))
test_data_ai_B = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/ai_embeded_subtaskA_test_multilingual.jsonl")))
val_data_ai_B = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/ai_embeded_subtaskA_dev_multilingual.jsonl")))
train_labels_ai_B = torch.ones(len(train_data_ai_B))
test_labels_ai_B = torch.ones(len(test_data_ai_B))
val_labels_ai_B = torch.ones(len(val_data_ai_B))


unsupervised_train_data_ai_B = train_data_ai_B[int(supervised_amt * len(train_data_human_A)):]
unsupervised_train_labels_ai_B = train_labels_ai_B[int(supervised_amt * len(train_data_human_A)):]
unsupervised_train_data_human_B = train_data_human_B[int(supervised_amt * len(train_data_human_A)):]
unsupervised_train_labels_human_B = train_labels_human_B[int(supervised_amt * len(train_data_human_A)):]

unsupervised_train_data_B = torch.cat((unsupervised_train_data_ai_B, unsupervised_train_data_human_B))
unsupervised_train_labels_B = torch.cat((unsupervised_train_labels_ai_B, unsupervised_train_labels_human_B))

unsupervised_train_data_B, unsupervised_train_labels_B = shuffle_data(unsupervised_train_data_B, unsupervised_train_labels_B)

test_data_B = torch.cat((test_data_human_B, test_data_ai_B), dim=0)
test_labels_B = torch.cat((test_labels_human_B, test_labels_ai_B), dim=0)
test_data_B, test_labels_B = shuffle_data(test_data_B, test_labels_B)



# unsupervised_train_data_A = torch.cat((unsupervised_train_data_human_A, unsupervised_train_data_ai_A), dim=0)
# unsupervised_train_labels_A = torch.cat((unsupervised_train_labels_human_A, unsupervised_train_labels_ai_A), dim=0)

# supervised_train_data_A = supervised_train_data_A[:int(supervised_percentage * len(supervised_train_data_A))]
# supervised_train_labels_A = supervised_train_labels_A[:int(supervised_percentage * len(supervised_train_labels_A))]


# data_train = data[:train_size]
# supervised_train_size = int(supervised_percentage * len(data_train))
# data_train_supervised = data_train[:supervised_train_size]
# data_train_unsupervised = data_train[supervised_train_size:]

# data_val = data[train_size:train_size + val_size]
# data_test = data[train_size + val_size:]

# labels_train = labels[:train_size]
# labels_train_supervised = labels_train[:supervised_train_size]
# labels_train_unsupervised = labels_train[supervised_train_size:]

# labels_val = labels[train_size:train_size+val_size]
# labels_test = labels[train_size+val_size:]


# validation_set = (data_val, labels_val)

model = ss_model()



for i in range(50):
    # ---------------
    # generate pseudo labels

    # Get prediction probabilities
    pred_probs = torch.tensor(model.predict(unsupervised_train_data_B))  # shape: (N, num_classes)

    # Get max probabilities and predicted labels
    max_probs, predicted_labels = torch.max(pred_probs, dim=1)

    # Filter for high-confidence predictions (>= 0.8)
    confident_mask = max_probs >= 0.8
    pseudo_labels = predicted_labels[confident_mask]
    data_pseudo_labeled = unsupervised_train_data_B[confident_mask]
    
    # Combine labeled and pseudo-labeled data
    data_train_combined = torch.concatenate([supervised_train_data_A, data_pseudo_labeled], axis=0)
    labels_train_combined = torch.concatenate([supervised_train_labels_A, pseudo_labels], axis=0)
    # ---------------

    # Train model again with both labeled and pseudo-labeled data
    model.fit(data_train_combined, labels_train_combined, epochs=10, validation_data=validation_set_A)

    evaluation = model.evaluate(test_data_B, test_labels_B)
    print(model.metrics_names)
    print(evaluation)

    if WANDB_ENABLED:
        wandb.log({
            "loss": evaluation[0],
            "accuracy": evaluation[1],
            "epoch": i
        })