import torch
import numpy as np

def load_path_in_tensor(path):
    return torch.tensor(np.array(torch.load(path)))

def get_dataset(dataset_name):
    if dataset_name == "Ghostbusters_all":
        train_data = load_path_in_tensor("Datasets/Ghostbusters_standardized_embedded/ghostbusters_ALL_train.jsonl")
        train_labels = load_path_in_tensor("Datasets/Ghostbusters_standardized_embedded/ghostbusters_ALL_train_labels.pt")

        val_data = load_path_in_tensor("Datasets/Ghostbusters_standardized_embedded/ghostbusters_ALL_val.jsonl")
        val_labels = load_path_in_tensor("Datasets/Ghostbusters_standardized_embedded/ghostbusters_ALL_val_labels.pt")

        test_data = load_path_in_tensor("Datasets/Ghostbusters_standardized_embedded/ghostbusters_ALL_test.jsonl")
        test_labels = load_path_in_tensor("Datasets/Ghostbusters_standardized_embedded/ghostbusters_ALL_test_labels.pt")
    elif dataset_name == "gpt_writing":
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
    elif dataset_name == "SemEval_complete":
        train_data = load_path_in_tensor("Datasets/SemEval_standardized_embedded/monolingual/monolingual_complete_train.jsonl")
        train_labels = load_path_in_tensor("Datasets/SemEval_standardized_embedded/monolingual/monolingual_complete_train_labels.pt")

        val_data = load_path_in_tensor("Datasets/SemEval_standardized_embedded/monolingual/monolingual_complete_val.jsonl")
        val_labels = load_path_in_tensor("Datasets/SemEval_standardized_embedded/monolingual/monolingual_complete_val_labels.pt")

        test_data = load_path_in_tensor("Datasets/SemEval_standardized_embedded/monolingual/monolingual_complete_test.jsonl")
        test_labels = load_path_in_tensor("Datasets/SemEval_standardized_embedded/monolingual/monolingual_complete_test_labels.pt")
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
