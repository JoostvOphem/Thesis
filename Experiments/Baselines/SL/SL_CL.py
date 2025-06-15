import tensorflow as tf
import torch
import numpy as np
import json

from data_utils import get_dataset


DATASET = "Ghostbusters_all"
TEST_DATASET = "gpt2"
ROBERTA_USED = "Ghostbusters_all"

WANDB_ENABLED = True
if WANDB_ENABLED:
    import wandb

    run = wandb.init(
        project="CL_testing",
        name=f'SL_CL_{DATASET}_{TEST_DATASET}_{ROBERTA_USED}'
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

train_data, train_labels, val_data, val_labels, test_data_A, test_labels_A = get_dataset(DATASET, ROBERTA_USED)
_, _, _, _, test_data_B, test_labels_B = get_dataset(TEST_DATASET, ROBERTA_USED)

# shuffle the data
def shuffle_data(data, labels):
    indices = torch.randperm(len(data))
    return data[indices], labels[indices]

# train_data, train_labels = shuffle_data(train_data, train_labels)
# val_data, val_labels = shuffle_data(val_data, val_labels)
# test_data, test_labels = shuffle_data(test_data, test_labels)

train_data = tf.nn.l2_normalize(train_data, axis=1)
val_data = tf.nn.l2_normalize(val_data, axis=1)
test_data_B = tf.nn.l2_normalize(test_data_B, axis=1)
test_data_A = tf.nn.l2_normalize(test_data_A, axis=1)

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
consistency_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.00001, weight_decay=0.01) # lr was 0.00001, weight decay was 0.01


for i in range(20):
    # ---------------
    model.fit(train_data, 
              train_labels, 
              epochs=9, 
              validation_data=validation_set,
              class_weight=class_weight_dict,
              batch_size=32,)
    
    # consistency training
    consistency_losses = consistency_training_step_advanced(
        model,
        train_data,
        consistency_weight=1,
        batch_size=64,
        epochs=1,
        shuffle=True,
        interpolation_pairs='random',
        optimizer=consistency_optimizer
    )

    logged_consistency_loss = tf.reduce_mean(consistency_losses).numpy()


    test_evaluation_B = model.evaluate(test_data_B, test_labels_B)
    test_evaluation_A = model.evaluate(test_data_A, test_labels_A)
    train_evaluation = model.evaluate(train_data, train_labels)

    # sanity check to see if the model is not just predicting the majority class
    # by checking to see if the validation set's first 10 predictions are not all the same
    # predictions = model.predict(validation_set[0])
    # predicted_labels = np.argmax(predictions, axis=1)
    # print(predicted_labels[:10])
    # print(val_labels[:10])

    # ---------------

    if WANDB_ENABLED:
        wandb.log({
            "consistency_loss": logged_consistency_loss,
            "train_loss": test_evaluation_A[0],
            "train_accuracy": test_evaluation_A[1],
            "test_loss": test_evaluation_B[0],
            "test_accuracy": test_evaluation_B[1],
            "epoch": i
        })
