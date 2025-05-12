from datasets import load_dataset
import pandas as pd
import tensorflow as tf
import torch
import numpy as np


WANDB_ENABLED = False
if WANDB_ENABLED:
    import wandb

    run = wandb.init(
        project="CL_Roberta_pseudolabeling",
        name='test_custom_loss'
    )

def difference_from_expected(preds, C):
  A, B = preds
  mid = (A+B)/2
  return mid-C

def consistency_loss(model, x_batch, reshape=(-1,768,1)):
    """
    Computes the consistency loss for a batch of embeddings x_batch.
    For each pair (A, B) in the batch, compares the average of predictions to the prediction of the average embedding.
    """
    batch_size = x_batch.shape[0]
    loss = 0.0
    count = 0
    print(batch_size, "entering for loop")
    # For efficiency, sample random pairs or use adjacent pairs
    for i in range(batch_size - 1):
        A = x_batch[i]
        B = x_batch[i+1]

        # Reshape for model input
        A_reshaped = A.reshape(1, *A.shape)
        B_reshaped = B.reshape(1, *B.shape)
        mix_reshaped = ((A+B)/2).reshape(1, *A.shape)
        # Model predictions
        pred_A = model(A_reshaped, training=True)
        pred_B = model(B_reshaped, training=True)

        pred_mix = model(mix_reshaped, training=True)
        avg_pred = (pred_A + pred_B) / 2
        loss += tf.reduce_mean(tf.square(avg_pred - pred_mix))

        count += 1
    if count == 0:
        return tf.constant(0.0)
    return loss / count

def custom_loss(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(
    y_true, y_pred, from_logits=False, ignore_class=None, axis=-1
    )

# data = torch.tensor(np.array(torch.load("subset_embeddings.npy")))
# labels = torch.load("subset_labels.npy")

def ss_model(loss_function = "sparse_categorical_crossentropy",
             optimizer = "AdamW",
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

# dataset_A

supervised_amt = 0.1

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

# # data = torch.tensor(np.array(torch.load("subset_embeddings.npy")))
# # labels = torch.load("subset_labels.npy")

# train_data_human = torch.tensor(
#     np.array(
#         torch.load("../SL/gpt-2-output-dataset-master/data/webtext-embedded.train")))
# test_data_human = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/webtext-embedded.test")))
# val_data_human =torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/webtext-embedded.valid")))
# train_labels_human = torch.zeros(len(train_data_human))
# test_labels_human = torch.zeros(len(test_data_human))
# val_labels_human = torch.zeros(len(val_data_human))

# train_data_ai = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/small-117M-embedded.train")))
# test_data_ai = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/small-117M-embedded.train")))
# val_data_ai = torch.tensor(np.array(torch.load("../SL/gpt-2-output-dataset-master/data/small-117M-embedded.train")))
# train_labels_ai = torch.ones(len(train_data_ai))
# test_labels_ai = torch.ones(len(test_data_ai))
# val_labels_ai = torch.ones(len(val_data_ai))

# # train, test, val = 0.6, 0.3, 0.1 # percentages
# supervised_percentage = 0.1

# # train_size = int(train * len(data))
# # val_size = int(val * len(data))

# supervised_train_data_human = train_data_human[:int(supervised_percentage * len(train_data_human))]
# supervised_train_labels_human = train_labels_human[:int(supervised_percentage * len(train_labels_human))]
# unsupervised_train_data_human = train_data_human[int(supervised_percentage * len(train_data_human)):]
# unsupervised_train_labels_human = train_labels_human[int(supervised_percentage * len(train_labels_human)):]

# supervised_train_data_ai = train_data_ai[:int(supervised_percentage * len(train_data_ai))]
# supervised_train_labels_ai = train_labels_ai[:int(supervised_percentage * len(train_labels_ai))]
# unsupervised_train_data_ai = train_data_ai[int(supervised_percentage * len(train_data_ai)):]
# unsupervised_train_labels_ai = train_labels_ai[int(supervised_percentage * len(train_labels_ai)):]

# supervised_train_data = torch.cat((supervised_train_data_human, supervised_train_data_ai), dim=0)
# supervised_train_labels = torch.cat((supervised_train_labels_human, supervised_train_labels_ai), dim=0)

# unsupervised_train_data = torch.cat((unsupervised_train_data_human, unsupervised_train_data_ai), dim=0)
# unsupervised_train_labels = torch.cat((unsupervised_train_labels_human, unsupervised_train_labels_ai), dim=0)

# val_data = torch.cat((val_data_human, val_data_ai), dim=0)
# val_labels = torch.cat((val_labels_human, val_labels_ai), dim=0)

# test_data = torch.cat((test_data_human, test_data_ai), dim=0)
# test_labels = torch.cat((test_labels_human, test_labels_ai), dim=0)

# # shuffle the data
# def shuffle_data(data, labels):
#     indices = torch.randperm(len(data))
#     return data[indices], labels[indices]

# supervised_train_data, supervised_train_labels = shuffle_data(supervised_train_data, supervised_train_labels)
# unsupervised_train_data, unsupervised_train_labels = shuffle_data(unsupervised_train_data, unsupervised_train_labels)
# val_data, val_labels = shuffle_data(val_data, val_labels)
# test_data, test_labels = shuffle_data(test_data, test_labels)

# validation_set = (val_data, val_labels)

model = ss_model()
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.1e-3)


for i in range(50):
    # ---------------
    # generate pseudo labels

    # Get prediction probabilities
    pred_probs = torch.tensor(model.predict(unsupervised_train_data_B))  # shape: (N, num_classes)

    # Get max probabilities and predicted labels
    max_probs, predicted_labels = torch.max(pred_probs, dim=1)

    # Filter for high-confidence predictions
    confidence_amt_to_miss = 0.21
    confident_mask_below = (max_probs <= confidence_amt_to_miss)
    confident_mask_above = (max_probs >= 1-confidence_amt_to_miss)

    confident_mask = confident_mask_below | confident_mask_above
    print(max_probs)
    print(confident_mask)
    # for thing in confident_mask:
    #     if thing:
    #         print("True")
    #     else:
    #         print("False")

    pseudo_labels = predicted_labels[confident_mask]
    data_pseudo_labeled = unsupervised_train_data_B[confident_mask]
    
    # Combine labeled and pseudo-labeled data
    data_train_combined = torch.concatenate([supervised_train_data_A, data_pseudo_labeled], axis=0)
    labels_train_combined = torch.concatenate([supervised_train_labels_A, pseudo_labels], axis=0)
    # ---------------

    # Shuffle combined data (optional, but good practice)
    indices = torch.randperm(len(data_train_combined))
    data_train_combined = data_train_combined[indices]
    labels_train_combined = labels_train_combined[indices]

    # Train model again with both labeled and pseudo-labeled data
    # model.fit(data_train_combined, labels_train_combined, epochs=1, validation_data=validation_set)

    # ---- for each batch (currently just one)
    x_batch = data_train_combined
    y_batch = labels_train_combined
    # y_batch = tf.one_hot(labels_train_combined, depth=2)
    with tf.GradientTape() as tape:
        # Supervised loss (optional, if you want to combine)
        pred = model(x_batch, training=True)
        supervised_loss = custom_loss(y_batch, pred)
        # Consistency loss
        c_loss = consistency_loss(model, x_batch)
        print("consistency loss: ", c_loss)
        # Combine losses if desired (here: only consistency loss)
        loss = supervised_loss + 0.1 * c_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # ---- end for each batch
    evaluation = model.evaluate(test_data_B, test_labels_B)

    if WANDB_ENABLED:
        wandb.log({
            "loss": evaluation[0],
            "accuracy": evaluation[1],
            "epoch": i
        })