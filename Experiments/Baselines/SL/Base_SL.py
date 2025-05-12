from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np


DATASET = "gpt-2"
WANDB_ENABLED = True
if WANDB_ENABLED:
    import wandb

    run = wandb.init(
        project="SL_Roberta",
        name=f'SemEval_more_data_{DATASET}'
    )

def s_model(loss_function = "sparse_categorical_crossentropy",
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

# def s_model(
#     loss_function="sparse_categorical_crossentropy",
#     optimizer="adamw",  # <-- try this instead of SGD
#     metrics=["accuracy"],
#     layers=[
#         tf.keras.layers.Input(shape=(768,1)),  # <-- flattened 768-dim input
#         tf.keras.layers.Dense(128, activation="relu"),
#         tf.keras.layers.Dense(64, activation="relu"),
#         tf.keras.layers.Dense(2, activation="softmax")
#     ]
# ):
#     model = tf.keras.models.Sequential(layers)
#     model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
#     return model

if DATASET == "SemEval":
    train_data_human = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/human_embeded_subtaskA_train_monolingual.jsonl")))
    test_data_human = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/human_embeded_subtaskA_test_monolingual.jsonl")))
    val_data_human =torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/human_embeded_subtaskA_dev_monolingual.jsonl")))
    train_data_ai = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/ai_embeded_subtaskA_train_monolingual.jsonl")))
    test_data_ai = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/ai_embeded_subtaskA_train_monolingual.jsonl")))
    val_data_ai = torch.tensor(np.array(torch.load("SemEval2024-M4/SubtaskA/ai_embeded_subtaskA_train_monolingual.jsonl")))
elif DATASET == "gpt-2":
    train_data_human = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/small-117M-embedded.test")))
    test_data_human = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/small-117M-embedded.test")))
    val_data_human = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/small-117M-embedded.valid")))
    train_data_ai = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/webtext-embedded.train")))
    test_data_ai = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/webtext-embedded.test")))
    val_data_ai = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/webtext-embedded.valid")))

train_labels_human = torch.zeros(len(train_data_human))
test_labels_human = torch.zeros(len(test_data_human))
val_labels_human = torch.zeros(len(val_data_human))
train_labels_ai = torch.ones(len(train_data_ai))
test_labels_ai = torch.ones(len(test_data_ai))
val_labels_ai = torch.ones(len(val_data_ai))

test_data = torch.cat((test_data_human, test_data_ai), dim=0)
test_labels = torch.cat((test_labels_human, test_labels_ai), dim=0)
train_data = torch.cat((train_data_human, train_data_ai), dim=0)
train_labels = torch.cat((train_labels_human, train_labels_ai), dim=0)
val_data = torch.cat((val_data_human, val_data_ai), dim=0)
val_labels = torch.cat((val_labels_human, val_labels_ai), dim=0)

# shuffle the data
def shuffle_data(data, labels):
    indices = torch.randperm(len(data))
    return data[indices], labels[indices]

train_data, train_labels = shuffle_data(train_data, train_labels)
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

model = s_model()

for i in range(50):
    # ---------------
    model.fit(train_data, train_labels, epochs=50, validation_data=validation_set)

    evaluation = model.evaluate(test_data, test_labels)

    if WANDB_ENABLED:
        wandb.log({
            "loss": evaluation[0],
            "accuracy": evaluation[1],
            "epoch": i
        })