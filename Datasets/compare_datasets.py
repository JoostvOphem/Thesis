import torch
from sentence_transformers import SentenceTransformer

import sys
sys.path.append('.')

from Datasets.dataset_utils import read_jsonl_dataset

# from torch.distributions import kl_divergence, Normal

def get_embedding(text):
    # Initialize model (this will be done only once)
    if not hasattr(get_embedding, "model"):
        get_embedding.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get embedding for the text
    embedding = get_embedding.model.encode(text, convert_to_tensor=True)
    return embedding

def calculate_ecdf(embeddings):
    n = embeddings.shape[0]
    ecdfs = []
    for dim in range(embeddings.shape[1]):
        sorted_values = torch.sort(embeddings[:, dim])[0]
        ecdf = torch.linspace(0, 1, n)
        ecdfs.append((sorted_values, ecdf))
    return ecdfs

# Calculate Wasserstein distance (Earth Mover's Distance) between ECDFs
# This is a more appropriate distance measure for empirical distributions
def wasserstein_distance(ecdf1, ecdf2):
    distances = []
    for (values1, cdf1), (values2, cdf2) in zip(ecdf1, ecdf2):
        # Interpolate CDFs to same points
        common_points = torch.sort(torch.cat([values1, values2]))[0]
        cdf1_interpolated = torch.searchsorted(values1, common_points) / len(values1)
        cdf2_interpolated = torch.searchsorted(values2, common_points) / len(values2)
        
        # Calculate Wasserstein distance
        distances.append(torch.abs(cdf1_interpolated - cdf2_interpolated).mean())
    
    return torch.mean(torch.stack(distances))


def get_difference_between_datasets(dataset1, dataset2, method='Wasserstein'):
    # Get embeddings for all texts in dataset1
    embeddings1 = []
    for text in dataset1['text']:
        embedding = get_embedding(text)
        embeddings1.append(embedding)
    embeddings1 = torch.stack(embeddings1)
    
    # Get embeddings for all texts in dataset2
    embeddings2 = []
    for text in dataset2['text']:
        embedding = get_embedding(text)
        embeddings2.append(embedding)
    embeddings2 = torch.stack(embeddings2)
    
    if method == 'Wasserstein':    
        ecdf1 = calculate_ecdf(embeddings1)
        ecdf2 = calculate_ecdf(embeddings2)
        return wasserstein_distance(ecdf1, ecdf2)

    elif method == 'Cosine':
        avg_embedding1 = torch.mean(embeddings1, dim=0)
        avg_embedding2 = torch.mean(embeddings2, dim=0)
        distance = 1 - torch.nn.functional.cosine_similarity(avg_embedding1, avg_embedding2, dim=0)
        return distance


# dataset1 = read_jsonl_dataset('Datasets/Ghostbusters_standardized/claude_complete.jsonl')
# dataset2 = read_jsonl_dataset('Datasets/Ghostbusters_standardized/gpt_complete.jsonl')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_distance_heatmap(distance_matrix, ai_types, location='Figures/ghostbusters_distance_heatmap.png'):
    """
    Create a heatmap visualization of dataset distances.
    
    Args:
    distance_matrix (numpy.ndarray): 2D array of distances between datasets
    ai_types (list): List of AI types corresponding to rows/columns of matrix
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(distance_matrix, 
                annot=True,  # Show numeric values
                cmap='YlGnBu',  # Color palette
                xticklabels=ai_types, 
                yticklabels=ai_types,
                cbar_kws={'label': 'Dataset Distance'}
    )
    
    plt.title('Dataset Distance Heatmap', fontsize=16)
    plt.xlabel('AI Types', fontsize=12)
    plt.ylabel('AI Types', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(location, dpi=300, bbox_inches='tight')
    plt.close()

def shrink_dataset(dataset, size=10):
    small_dataset = {lst: [] for lst in dataset.keys()}

    for i in range(size):
        for lst in dataset.keys():
            small_dataset[lst].append(dataset[lst][i])
    return small_dataset


def compare_datasets_with_themselves(
        standardized_dataset_path='Datasets/Ghostbusters_standardized', # standard for ghostbusters
        AI_types = ['claude', 'gpt', 'gpt_prompt1', 'gpt_prompt2', 'gpt_semantic', 'gpt_writing'], # AI types to compare, standard is all for ghostbusters
        version='complete', # versions: complete, dev, test, train
        heatmap_location='Figures/ghostbusters_distance_heatmap.png', # location to save the heatmap,
        method='Wasserstein' # method to use for distance calculation, options: 'Wasserstein', 'Cosine'
        ):
    
    distance_matrix = torch.zeros(len(AI_types), len(AI_types))

    for i, AI_type in enumerate(AI_types):
        dataset1 = read_jsonl_dataset(f'{standardized_dataset_path}/{AI_type}_{version}.jsonl')
        dataset1 = shrink_dataset(dataset1, 100)

        for j, AI_type2 in enumerate(AI_types):
            if i > j: # no need to calculate twice
                continue
            if i == j:
                distance_matrix[i, j] = 0 
                # if same dataset, distance is 0 (also true when actually running the code. This just reduces the number of calculations)
                continue
            dataset2 = read_jsonl_dataset(f'{standardized_dataset_path}/{AI_type2}_{version}.jsonl')
            dataset2 = shrink_dataset(dataset2, 100)

            distance = get_difference_between_datasets(dataset1, dataset2, method=method)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

            print(AI_type, AI_type2, distance.item())
    
    # Create a heatmap of the distances
    print("Distance matrix:")
    print(distance_matrix)
    create_distance_heatmap(distance_matrix.numpy(), AI_types, location=heatmap_location)

def compare_datasets_with_eachother(
        dataset_paths,
        AI_types_list,
        version='complete', # versions: complete, dev, test, train
        heatmap_location='Figures/meta-heatmap.png', # location to save the heatmap
        method='Wasserstein' # method to use for distance calculation, options: 'Wasserstein', 'Cosine'
        ):
    
    all_AI_types = []
    all_paths = []
    for AI_types in AI_types_list:
        all_AI_types.extend(AI_types)

    for (dataset_path, AI_types) in zip(dataset_paths, AI_types_list):
        for AI_type in AI_types:
            all_paths.append(f'{dataset_path}/{AI_type}_{version}.jsonl')
    
    distance_matrix = torch.zeros(len(all_paths), len(all_paths))
    
    for i, path1 in enumerate(all_paths):
        for j, path2 in enumerate(all_paths):
            if i > j: # no need to calculate twice
                continue
            if i == j:
                distance_matrix[i, j] = 0 
                # if same dataset, distance is 0 (also true when actually running the code. This just reduces the number of calculations)
                continue
            dataset1 = read_jsonl_dataset(path1)
            dataset1 = shrink_dataset(dataset1, 10)

            dataset2 = read_jsonl_dataset(path2)
            dataset2 = shrink_dataset(dataset2, 10)

            distance = get_difference_between_datasets(dataset1, dataset2, method=method)
            distance_matrix[i,j] = distance
            distance_matrix[j,i] = distance
            print(path1, path2, distance.item())

    create_distance_heatmap(distance_matrix.numpy(), all_AI_types, location=heatmap_location)
    return

    
if __name__ == "__main__":
    method = 'Cosine'
    version = 'complete'

    # compare_datasets_with_themselves(
    #     standardized_dataset_path='Datasets/Ghostbusters_standardized',
    #     AI_types=['claude', 'gpt', 'gpt_prompt1', 'gpt_prompt2', 'gpt_semantic', 'gpt_writing'],
    #     version=version,
    #     heatmap_location=f'Figures/ghostbusters_{method}_distance_heatmap.png',
    #     method=method
    # )

    # compare_datasets_with_themselves(
    #     standardized_dataset_path='Datasets/SemEval_standardized/monolingual',
    #     AI_types=['monolingual_'+ai_type for ai_type in ['bloomz', 'chatGPT', 'cohere', 'complete', 'davinci', 'dolly', 'GPT4']],
    #     version=version,
    #     heatmap_location=f'Figures/monolingual_{method}_distance_heatmap.png',
    #     method=method
    # )

    # compare_datasets_with_themselves(
    #     standardized_dataset_path='Datasets/SemEval_standardized/multilingual',
    #     AI_types=['multilingual_'+ai_type for ai_type in ['bloomz', 'chatGPT', 'cohere', 'complete', 'davinci', 'dolly', 'GPT4']],
    #     version=version,
    #     heatmap_location=f'Figures/multilingual_{method}_distance_heatmap.png',
    #     method=method
    # )

    # compare_datasets_with_eachother(
    #     dataset_paths=[
    #         'Datasets/Ghostbusters_standardized',
    #         'Datasets/SemEval_standardized/monolingual',
    #         'Datasets/SemEval_standardized/multilingual'
    #     ],
    #     AI_types_list=[
    #         ['claude', 'gpt', 'gpt_prompt1', 'gpt_prompt2', 'gpt_semantic', 'gpt_writing'],
    #         ['monolingual_'+ai_type for ai_type in ['bloomz', 'chatGPT', 'cohere', 'complete', 'davinci', 'dolly', 'GPT4']],
    #         ['multilingual_'+ai_type for ai_type in ['bloomz', 'chatGPT', 'cohere', 'complete', 'davinci', 'dolly', 'GPT4']]
    #     ],
    #     version=version,
    #     heatmap_location=f'Figures/meta_{method}-heatmap.png',
    #     method=method
    # )
    pass