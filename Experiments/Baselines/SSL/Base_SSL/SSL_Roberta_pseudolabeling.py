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
        name='Roberta-Fixed'
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
supervised_percentage = 0.1

# train_size = int(train * len(data))
# val_size = int(val * len(data))
train_data_human = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/human_embeded_subtaskA_train_multilingual.jsonl")))
test_data_human = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/human_embeded_subtaskA_train_multilingual.jsonl")))
val_data_human =torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/webtext-embedded.valid")))
train_labels_human = torch.zeros(len(train_data_human))
test_labels_human = torch.zeros(len(test_data_human))
val_labels_human = torch.zeros(len(val_data_human))

train_data_ai = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/small-117M-embedded.train")))
test_data_ai = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/small-117M-embedded.train")))
val_data_ai = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/small-117M-embedded.train")))
train_labels_ai = torch.ones(len(train_data_ai))
test_labels_ai = torch.ones(len(test_data_ai))
val_labels_ai = torch.ones(len(val_data_ai))
supervised_train_data_human = train_data_human[:int(supervised_percentage * len(train_data_human))]
supervised_train_labels_human = train_labels_human[:int(supervised_percentage * len(train_labels_human))]
unsupervised_train_data_human = train_data_human[int(supervised_percentage * len(train_data_human)):]
unsupervised_train_labels_human = train_labels_human[int(supervised_percentage * len(train_labels_human)):]

supervised_train_data_ai = train_data_ai[:int(supervised_percentage * len(train_data_ai))]
supervised_train_labels_ai = train_labels_ai[:int(supervised_percentage * len(train_labels_ai))]
unsupervised_train_data_ai = train_data_ai[int(supervised_percentage * len(train_data_ai)):]
unsupervised_train_labels_ai = train_labels_ai[int(supervised_percentage * len(train_labels_ai)):]

supervised_train_data = torch.cat((supervised_train_data_human, supervised_train_data_ai), dim=0)
supervised_train_labels = torch.cat((supervised_train_labels_human, supervised_train_labels_ai), dim=0)

unsupervised_train_data = torch.cat((unsupervised_train_data_human, unsupervised_train_data_ai), dim=0)
unsupervised_train_labels = torch.cat((unsupervised_train_labels_human, unsupervised_train_labels_ai), dim=0)

val_data = torch.cat((val_data_human, val_data_ai), dim=0)
val_labels = torch.cat((val_labels_human, val_labels_ai), dim=0)

test_data = torch.cat((test_data_human, test_data_ai), dim=0)
test_labels = torch.cat((test_labels_human, test_labels_ai), dim=0)

# shuffle the data
def shuffle_data(data, labels):
    indices = torch.randperm(len(data))
    return data[indices], labels[indices]

supervised_train_data, supervised_train_labels = shuffle_data(supervised_train_data, supervised_train_labels)
unsupervised_train_data, unsupervised_train_labels = shuffle_data(unsupervised_train_data, unsupervised_train_labels)
val_data, val_labels = shuffle_data(val_data, val_labels)
test_data, test_labels = shuffle_data(test_data, test_labels)

validation_set = (val_data, val_labels)


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
    pred_probs = torch.tensor(model.predict(unsupervised_train_data))  # shape: (N, num_classes)

    # Get max probabilities and predicted labels
    max_probs, predicted_labels = torch.max(pred_probs, dim=1)

    # Filter for high-confidence predictions (>= 0.8)
    confident_mask = max_probs >= 0.8
    pseudo_labels = predicted_labels[confident_mask]
    data_pseudo_labeled = unsupervised_train_data[confident_mask]
    
    # Combine labeled and pseudo-labeled data
    data_train_combined = torch.concatenate([supervised_train_data, data_pseudo_labeled], axis=0)
    labels_train_combined = torch.concatenate([supervised_train_labels, pseudo_labels], axis=0)
    # ---------------

    # Train model again with both labeled and pseudo-labeled data
    model.fit(data_train_combined, labels_train_combined, epochs=10, validation_data=validation_set)

    evaluation = model.evaluate(test_data, test_labels)
    print(model.metrics_names)
    print(evaluation)

    if WANDB_ENABLED:
        wandb.log({
            "loss": evaluation[0],
            "accuracy": evaluation[1],
            "epoch": i
        })