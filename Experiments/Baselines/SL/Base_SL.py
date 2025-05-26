from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np
import json
import scipy


DATASET = "monolingual_davinci"  # options: "gpt_writing", "monolingual_davinci"
WANDB_ENABLED = True
if WANDB_ENABLED:
    import wandb

    run = wandb.init(
        project="SL_Roberta_compare",
        name=f'26-05_{DATASET}'
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

if DATASET == "gpt_writing":
    train_data = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_train.jsonl")))
    train_labels = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_train_labels.pt")))

    val_data = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_val.jsonl")))
    val_labels = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_val_labels.pt")))

    test_data = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_test.jsonl")))
    test_labels = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_test_labels.pt")))

if DATASET == "monolingual_davinci":
    train_data = torch.tensor(np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_train.jsonl")))
    train_labels = torch.tensor(np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_train_labels.pt")))

    val_data = torch.tensor(np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_val.jsonl")))
    val_labels = torch.tensor(np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_val_labels.pt")))

    test_data = torch.tensor(np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_test.jsonl")))
    test_labels = torch.tensor(np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_test_labels.pt")))


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
            "loss": evaluation[0],
            "accuracy": evaluation[1],
            "train_loss": train_evaluation[0],
            "train_accuracy": train_evaluation[1],
            "epoch": i
        })

# from datasets import load_dataset
# import pandas as pd
# import tensorflow as tf
# import torch
# import numpy as np
# import json
# from sklearn.utils.class_weight import compute_class_weight

# DATASET = "monolingual_davinci"
# WANDB_ENABLED = False
# if WANDB_ENABLED:
#     import wandb
#     run = wandb.init(
#         project="SL_Roberta_compare",
#         name=f'SemEval_more_data_{DATASET}'
#     )

# def s_model(loss_function="sparse_categorical_crossentropy",
#              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#              metrics=["accuracy"],
#              epochs=30,
#              layers=None,
#              class_weights=None):

#     if layers is None:
#         layers = [
#             tf.keras.layers.Flatten(input_shape=[768,1], name="inputLayer"),
#             tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
#             tf.keras.layers.BatchNormalization(),  # Add batch normalization
#             tf.keras.layers.Dropout(0.3),
#             tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Dropout(0.3),
#             tf.keras.layers.Dense(10, activation="relu", name="hiddenlayer3"),
#             tf.keras.layers.Dense(2, activation="softmax", name="outputlayer")
#         ]

#     model = tf.keras.models.Sequential(layers)
    
#     model.compile(loss=loss_function,
#                  optimizer=optimizer,
#                  metrics=metrics)

#     return model

# def label_data(path):
#     labels = []
#     with open(path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 data = json.loads(line)
#                 labels.append(data["label"])
#     return np.array(labels, dtype=np.int32)

# def load_embedded_data(path):
#     data = torch.load(path)
#     data_array = np.array(data)
#     print(f"Loaded data shape: {data_array.shape}")
    
#     if len(data_array.shape) == 2:
#         data_array = data_array.reshape(data_array.shape[0], data_array.shape[1], 1)
    
#     return data_array

# def balance_dataset(data, labels, method='undersample'):
#     """Balance the dataset using different strategies"""
#     unique_labels, counts = np.unique(labels, return_counts=True)
#     print(f"Original distribution: {dict(zip(unique_labels, counts))}")
    
#     if method == 'undersample':
#         # Undersample majority class
#         min_count = np.min(counts)
#         balanced_data = []
#         balanced_labels = []
        
#         for label in unique_labels:
#             label_indices = np.where(labels == label)[0]
#             selected_indices = np.random.choice(label_indices, min_count, replace=False)
#             balanced_data.extend(data[selected_indices])
#             balanced_labels.extend(labels[selected_indices])
        
#         balanced_data = np.array(balanced_data)
#         balanced_labels = np.array(balanced_labels)
        
#     elif method == 'oversample':
#         # Oversample minority class
#         max_count = np.max(counts)
#         balanced_data = []
#         balanced_labels = []
        
#         for label in unique_labels:
#             label_indices = np.where(labels == label)[0]
#             if len(label_indices) < max_count:
#                 # Oversample with replacement
#                 selected_indices = np.random.choice(label_indices, max_count, replace=True)
#             else:
#                 selected_indices = label_indices
            
#             balanced_data.extend(data[selected_indices])
#             balanced_labels.extend(labels[selected_indices])
        
#         balanced_data = np.array(balanced_data)
#         balanced_labels = np.array(balanced_labels)
    
#     # Shuffle the balanced dataset
#     indices = np.random.permutation(len(balanced_data))
#     balanced_data = balanced_data[indices]
#     balanced_labels = balanced_labels[indices]
    
#     unique_labels_new, counts_new = np.unique(balanced_labels, return_counts=True)
#     print(f"New distribution: {dict(zip(unique_labels_new, counts_new))}")
    
#     return balanced_data, balanced_labels

# # Custom focal loss to handle class imbalance
# def focal_loss(alpha=0.25, gamma=2.0):
#     def focal_loss_fixed(y_true, y_pred):
#         # Convert to one-hot if needed
#         y_true = tf.cast(y_true, tf.int32)
#         y_true_one_hot = tf.one_hot(y_true, depth=2)
        
#         # Compute focal loss
#         ce = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)
#         p = tf.exp(-ce)
#         loss = alpha * (1 - p) ** gamma * ce
#         return tf.reduce_mean(loss)
#     return focal_loss_fixed

# amt = 2000

# # Load data
# train_data = load_embedded_data("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_train.jsonl")[:amt]
# train_labels = (np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_train_labels.pt")))[:amt]

# print("train")

# val_data = load_embedded_data("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_val_labels.pt")[:amt]
# val_labels = (np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_val_labels.pt")))[:amt]

# print("val")

# test_data = load_embedded_data("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_test.jsonl")[:amt]
# test_labels = (np.array(torch.load("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_test_labels.pt")))[:amt]

# # SOLUTION 1: Balance the training data
# # print("Balancing training data...")
# # train_data, train_labels = balance_dataset(train_data, train_labels, method='undersample')

# # SOLUTION 2: Compute class weights

# def get_class_weights(labels):
#     """Compute class weights for imbalanced datasets."""
#     class_amts = {}
#     for label in labels:
#         label = label.item()
#         if label not in class_amts:
#             class_amts[label] = 1
#         else:
#             class_amts[label] += 1
    
#     total_amt = len(labels)
#     print(class_amts)
#     class_weights = {i: class_amts[label] / total_amt for i, label in enumerate(class_amts)}
#     return class_weights


# class_weight_dict = get_class_weights(train_labels)
# print(f"Class weights: {class_weight_dict}")

# # Adjust model input shape
# actual_input_shape = train_data.shape[1:]
# print(f"Model input shape: {actual_input_shape}")

# # SOLUTION 3: Use focal loss instead of regular loss
# model_layers = [
#     tf.keras.layers.Flatten(input_shape=actual_input_shape, name="inputLayer"),
#     tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.4),  # Increased dropout
#     tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(10, activation="relu", name="hiddenlayer3"),
#     tf.keras.layers.Dense(2, activation="softmax", name="outputlayer")
# ]

# # Try different approaches:

# # APPROACH 1: Balanced data + class weights + regular loss
# print("\n" + "="*50)
# print("APPROACH 1: Balanced data + Class weights")
# print("="*50)

# model1 = s_model(
#     layers=model_layers,
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)  # Lower learning rate
# )

# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', 
#     patience=10, 
#     restore_best_weights=True
# )

# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss', 
#     factor=0.5, 
#     patience=5, 
#     min_lr=1e-7
# )

# print(train_data.shape, train_labels.shape)

# history1 = model1.fit(
#     train_data, train_labels,
#     epochs=100,
#     validation_data=(val_data, val_labels),
#     callbacks=[early_stopping, reduce_lr],
#     batch_size=32,
#     class_weight=class_weight_dict,  # Use class weights
#     verbose=1
# )

# evaluation1 = model1.evaluate(test_data, test_labels, verbose=1)
# print(f"Approach 1 - Test accuracy: {evaluation1[1]:.4f}")

# # APPROACH 2: Original imbalanced data + focal loss
# print("\n" + "="*50)
# print("APPROACH 2: Focal Loss (original data)")
# print("="*50)

# # Reload original imbalanced data
# train_data_orig = load_embedded_data("Datasets/SemEval_standardized_embedded/monolingual/monolingual_davinci_train.jsonl")[:amt]
# train_labels_orig = label_data("Datasets/SemEval_standardized/monolingual/monolingual_davinci_train.jsonl")[:amt]

# model2 = s_model(
#     layers=model_layers,
#     loss_function=focal_loss(alpha=0.75, gamma=2.0),  # Focal loss with higher alpha for minority class
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
# )

# history2 = model2.fit(
#     train_data_orig, train_labels_orig,
#     epochs=100,
#     validation_data=(val_data, val_labels),
#     callbacks=[early_stopping, reduce_lr],
#     batch_size=32,
#     verbose=1
# )

# evaluation2 = model2.evaluate(test_data, test_labels, verbose=1)
# print(f"Approach 2 - Test accuracy: {evaluation2[1]:.4f}")

# # Test both models with the prediction checker
# print("\n" + "="*50)
# print("TESTING APPROACH 1 (Balanced + Class Weights)")
# print("="*50)

# # You'll need to add your check_model_predictions function here
# # results1 = check_model_predictions(model1, test_data, test_labels)

# print("\n" + "="*50)
# print("TESTING APPROACH 2 (Focal Loss)")
# print("="*50)

# # results2 = check_model_predictions(model2, test_data, test_labels)

# if WANDB_ENABLED:
#     wandb.log({
#         "approach1_test_accuracy": evaluation1[1],
#         "approach2_test_accuracy": evaluation2[1]
#     })