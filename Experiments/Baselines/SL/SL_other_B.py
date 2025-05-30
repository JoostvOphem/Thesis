from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np
import json
import scipy

from data_utils import get_dataset


DATASET = "Ghostbusters_all"  # options: "gpt_writing", "monolingual_davinci", "GPT2", "Ghostbusters_all", "SemeVal_complete"
TEST_DATASET = "SemeVal_complete"
WANDB_ENABLED = True
if WANDB_ENABLED:
    import wandb

    run = wandb.init(
        project="30-05-comparisons",
        name=f'SL_B_{DATASET}_{TEST_DATASET}'
    )

def s_model(loss_function = "sparse_categorical_crossentropy",
             optimizer = tf.keras.optimizers.AdamW(
                 learning_rate=0.0005,
                 weight_decay=0.01),
             metrics=["accuracy"],
             epochs=30,
            #  layers = [tf.keras.layers.Flatten(input_shape=[768,1], name="inputLayer"),
            #            tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
            #            tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
            #            tf.keras.layers.Dense(10, activation="relu", name="hiddenlayer3"),
            #            tf.keras.layers.Dense(2, activation="softmax", name="outputlayer")]
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


def label_data(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data = json.loads(line)  # Parse each line as JSON
                labels.append(data["label"])
    
    # Convert to tensor
    return torch.tensor(labels, dtype=np.int32)

def load_path_in_tensor(path):
    return torch.tensor(np.array(torch.load(path)))

train_data, train_labels, val_data, val_labels, _, _ = get_dataset(DATASET)
_, _, _, _, test_data, test_labels = get_dataset(TEST_DATASET)

# shuffle the data
def shuffle_data(data, labels):
    indices = torch.randperm(len(data))
    return data[indices], labels[indices]

# train_data, train_labels = shuffle_data(train_data, train_labels)
# val_data, val_labels = shuffle_data(val_data, val_labels)
# test_data, test_labels = shuffle_data(test_data, test_labels)

train_data = tf.nn.l2_normalize(train_data, axis=1)
val_data = tf.nn.l2_normalize(val_data, axis=1)
test_data = tf.nn.l2_normalize(test_data, axis=1)

validation_set = (val_data, val_labels)

def get_class_weights(labels):
    """Compute class weights for imbalanced datasets."""
    class_amts = {}
    for label in labels:
        label = label.item()
        if label not in class_amts:
            class_amts[label] = 1
        else:
            class_amts[label] += 1
    
    total_amt = len(labels)
    print(class_amts)
    class_weights = {i: class_amts[label] / total_amt for i, label in enumerate(class_amts)}
    return class_weights
class_weight_dict = get_class_weights(train_labels)
print(class_weight_dict)

model = s_model()

for i in range(20):
    # ---------------
    model.fit(train_data, 
              train_labels, 
              epochs=20, 
              validation_data=validation_set,
              class_weight=class_weight_dict,
              batch_size=32,)

    evaluation = model.evaluate(test_data, test_labels)
    train_evaluation = model.evaluate(train_data, train_labels)

    # sanity check to see if the model is not just predicting the majority class
    # by checking to see if the validation set's first 10 predictions are not all the same
    predictions = model.predict(validation_set[0])
    predicted_labels = np.argmax(predictions, axis=1)
    print(predicted_labels[:10])
    print(val_labels[:10])

    # ---------------

    if WANDB_ENABLED:
        wandb.log({
            "test_loss": evaluation[0],
            "test_accuracy": evaluation[1],
            "train_loss": train_evaluation[0],
            "train_accuracy": train_evaluation[1],
            "epoch": i
        })
