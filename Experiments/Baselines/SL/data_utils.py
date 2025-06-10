import torch
import numpy as np

def load_path_in_tensor(path):
    return torch.tensor(np.array(torch.load(path)))

def get_dataset(dataset_name, roberta_used):

    if roberta_used.startswith("monolingual"):
        start_of_path = f"Datasets/SemEval_standardized_embedded/monolingual/roberta_{roberta_used}"
    elif roberta_used.startswith("multilingual"):
        start_of_path = f"Datasets/SemEval_standardized_embedded/multilingual/roberta_{roberta_used}"
    elif roberta_used.startswith("GPT2"):
        start_of_path = f"Datasets/GPT2_standardized_embedded/roberta_{roberta_used}"
    else:
        start_of_path = f"Datasets/Ghostbusters_standardized_embedded/roberta_{roberta_used}"

    train_data = load_path_in_tensor(f"{start_of_path}/{dataset_name}_train.jsonl")
    train_labels = load_path_in_tensor(f"{start_of_path}/{dataset_name}_train_labels.pt")

    val_data = load_path_in_tensor(f"{start_of_path}/{dataset_name}_val.jsonl")
    val_labels = load_path_in_tensor(f"{start_of_path}/{dataset_name}_val_labels.pt")

    test_data = load_path_in_tensor(f"{start_of_path}/{dataset_name}_test.jsonl")
    test_labels = load_path_in_tensor(f"{start_of_path}/{dataset_name}_test_labels.pt")

    return train_data, train_labels, val_data, val_labels, test_data, test_labels
