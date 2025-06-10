import torch
from sentence_transformers import SentenceTransformer

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import jensenshannon


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


def tfidf_distance(texts1, texts2):
    """
    Calculate distance between two text datasets using TF-IDF representations.
    
    Returns a value between 0 and 1, where 0 means identical and 1 means completely different.
    """
    # Combine all texts for fitting the vectorizer
    all_texts = texts1 + texts2
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split the matrix back into the two datasets
    tfidf_texts1 = tfidf_matrix[:len(texts1)]
    tfidf_texts2 = tfidf_matrix[len(texts1):]
    
    # Calculate the centroid (average) of each dataset's TF-IDF vectors
    centroid1 = np.asarray(tfidf_texts1.mean(axis=0))
    centroid2 = np.asarray(tfidf_texts2.mean(axis=0))
    
    # Calculate cosine similarity between centroids
    similarity = cosine_similarity(centroid1, centroid2)[0][0]
    
    # Convert to distance (1 - similarity)
    distance = 1.0 - similarity
    
    return distance

def vocabulary_overlap_distance(texts1, texts2):
    """
    Calculate distance based on vocabulary overlap between two text datasets.
    
    Returns a value between 0 and 1, where 0 means complete overlap and 1 means no overlap.
    """
    # Tokenize texts (simple whitespace splitting for demonstration)
    # For better results, consider using NLTK, spaCy, or a similar library
    tokens1 = set()
    tokens2 = set()
    
    for text in texts1:
        tokens1.update(text.lower().split())
    
    for text in texts2:
        tokens2.update(text.lower().split())
    
    # Calculate Jaccard distance (1 - Jaccard similarity)
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    if union == 0:  # Handle edge case of empty sets
        return 0.0
        
    jaccard_similarity = intersection / union
    jaccard_distance = 1.0 - jaccard_similarity
    
    return jaccard_distance

def topic_model_distance(texts1, texts2, n_topics=10):
    """
    Calculate distance between topic distributions of two text datasets.
    
    Uses LDA for topic modeling and Jensen-Shannon divergence to compare distributions.
    Returns a value between 0 and 1, where 0 means identical distributions.
    """
    # Create a TF vectorizer for LDA input
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', use_idf=False)
    
    # Combine texts for fitting the vectorizer
    all_texts = texts1 + texts2
    tf_matrix = vectorizer.fit_transform(all_texts)
    
    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tf_matrix)
    
    # Get topic distributions for each dataset
    tf_texts1 = vectorizer.transform(texts1)
    tf_texts2 = vectorizer.transform(texts2)
    
    topic_dist1 = lda.transform(tf_texts1)
    topic_dist2 = lda.transform(tf_texts2)
    
    # Calculate average topic distribution for each dataset
    avg_topic_dist1 = topic_dist1.mean(axis=0)
    avg_topic_dist2 = topic_dist2.mean(axis=0)
    
    # Ensure distributions sum to 1
    avg_topic_dist1 = avg_topic_dist1 / avg_topic_dist1.sum()
    avg_topic_dist2 = avg_topic_dist2 / avg_topic_dist2.sum()
    
    # Calculate Jensen-Shannon divergence between topic distributions
    js_distance = jensenshannon(avg_topic_dist1, avg_topic_dist2)
    
    return js_distance

def get_embeddings(texts):
    embeddings = []
    for text in texts:
        embedding = get_embedding(text)
        embeddings.append(embedding)
    return torch.stack(embeddings)


def get_difference_between_datasets(dataset1, dataset2, method='Wasserstein'):    
    texts1 = dataset1['text']
    texts2 = dataset2['text']
    
    if method == 'Wasserstein': 
        embeddings1 = get_embeddings(texts1)
        embeddings2 = get_embeddings(texts2)  
        ecdf1 = calculate_ecdf(embeddings1)
        ecdf2 = calculate_ecdf(embeddings2)
        return wasserstein_distance(ecdf1, ecdf2)

    elif method == 'Cosine':
        embeddings1 = get_embeddings(texts1)
        embeddings2 = get_embeddings(texts2)
        avg_embedding1 = torch.mean(embeddings1, dim=0)
        avg_embedding2 = torch.mean(embeddings2, dim=0)
        distance = 1 - torch.nn.functional.cosine_similarity(avg_embedding1, avg_embedding2, dim=0)
        return distance
    
    elif method == 'Vocabulary_Overlap':
        return torch.Tensor([vocabulary_overlap_distance(texts1, texts2)])
    
    elif method == 'Topic_Model':
        return torch.Tensor([topic_model_distance(texts1, texts2)])
    
    elif method == 'TF-IDF':
        return torch.Tensor([tfidf_distance(texts1, texts2)])


# dataset1 = read_jsonl_dataset('Datasets/Ghostbusters_standardized/claude_complete.jsonl')
# dataset2 = read_jsonl_dataset('Datasets/Ghostbusters_standardized/gpt_complete.jsonl')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_distance_heatmap(distance_matrix, ai_types, location='Figures/ghostbusters_distance_heatmap.png', numbers=True):
    """
    Create a heatmap visualization of dataset distances.
    
    Args:
    distance_matrix (numpy.ndarray): 2D array of distances between datasets
    ai_types (list): List of AI types corresponding to rows/columns of matrix
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(distance_matrix, 
                annot=numbers,  # Show numeric values
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
    file_extension = location.split('.')[-1]
    plt.savefig(location, format=file_extension, bbox_inches='tight')
    plt.close()

def shrink_dataset(dataset, size=100):
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
        method='Wasserstein', # method to use for distance calculation, options: 'Wasserstein', 'Cosine',
        numbers=True # whether to show numbers on the heatmap
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
    create_distance_heatmap(distance_matrix.numpy(), AI_types, location=heatmap_location, numbers=numbers)

def compare_datasets_with_eachother(
    dataset_paths,
    AI_types_list,
    version='complete', # versions: complete, dev, test, train
    heatmap_location='Figures/meta-heatmap.png', # location to save the heatmap
    method='Wasserstein', # method to use for distance calculation, options: 'Wasserstein', 'Cosine'
    numbers=True
):
    all_AI_types = []
    all_paths = []
    dataset_indices = []  # Track which dataset each AI type belongs to
    
    for dataset_idx, AI_types in enumerate(AI_types_list):
        all_AI_types.extend(AI_types)
        dataset_indices.extend([dataset_idx] * len(AI_types))
        
    for (dataset_path, AI_types) in zip(dataset_paths, AI_types_list):
        for AI_type in AI_types:
            all_paths.append(f'{dataset_path}/{AI_type}_{version}.jsonl')
    
    distance_matrix = torch.zeros(len(all_paths), len(all_paths))
    
    # First pass: calculate all pairwise distances
    for i, path1 in enumerate(all_paths):
        for j, path2 in enumerate(all_paths):
            if i > j: # no need to calculate twice
                continue
            if i == j:
                distance_matrix[i, j] = 0
                continue
                
            dataset1 = read_jsonl_dataset(path1)
            dataset1 = shrink_dataset(dataset1, 100)
            dataset2 = read_jsonl_dataset(path2)
            dataset2 = shrink_dataset(dataset2, 100)
            distance = get_difference_between_datasets(dataset1, dataset2, method=method)
            distance_matrix[i,j] = distance
            distance_matrix[j,i] = distance
            print(path1, path2, distance.item())
    
    # Second pass: calculate dataset averages for lower triangle
    num_datasets = len(dataset_paths)
    dataset_avg_matrix = torch.zeros(num_datasets, num_datasets)
    
    # Calculate average distances between datasets
    for dataset_i in range(num_datasets):
        for dataset_j in range(num_datasets):
            # Find all AI types for each dataset
            indices_i = [k for k, d in enumerate(dataset_indices) if d == dataset_i]
            indices_j = [k for k, d in enumerate(dataset_indices) if d == dataset_j]
            
            # Calculate average distance between all pairs (excluding same AI type comparisons)
            distances = []
            for idx_i in indices_i:
                for idx_j in indices_j:
                    # Skip comparisons where the AI type names are the same
                    if all_AI_types[idx_i] != all_AI_types[idx_j]:
                        distances.append(distance_matrix[idx_i, idx_j].item())
            
            if len(distances) > 0:
                dataset_avg_matrix[dataset_i, dataset_j] = sum(distances) / len(distances)
            else:
                dataset_avg_matrix[dataset_i, dataset_j] = 0
    
    # Modify distance_matrix: keep upper triangle as is, replace lower triangle with dataset averages
    for i in range(len(all_paths)):
        for j in range(len(all_paths)):
            if i > j:  # Lower triangle
                dataset_i = dataset_indices[i]
                dataset_j = dataset_indices[j]
                distance_matrix[i, j] = dataset_avg_matrix[dataset_i, dataset_j]
    
    create_distance_heatmap(distance_matrix.numpy(), all_AI_types, location=heatmap_location, numbers=numbers)
# def compare_datasets_with_eachother(
#     dataset_paths,
#     AI_types_list,
#     version='complete', # versions: complete, dev, test, train
#     heatmap_location='Figures/meta-heatmap.png', # location to save the heatmap
#     method='Wasserstein', # method to use for distance calculation, options: 'Wasserstein', 'Cosine'
#     numbers=True
# ):
#     all_AI_types = []
#     all_paths = []
#     dataset_indices = []  # Track which dataset each AI type belongs to
    
#     for dataset_idx, AI_types in enumerate(AI_types_list):
#         all_AI_types.extend(AI_types)
#         dataset_indices.extend([dataset_idx] * len(AI_types))
        
#     for (dataset_path, AI_types) in zip(dataset_paths, AI_types_list):
#         for AI_type in AI_types:
#             all_paths.append(f'{dataset_path}/{AI_type}_{version}.jsonl')
    
#     distance_matrix = torch.zeros(len(all_paths), len(all_paths))
    
#     # First pass: calculate all pairwise distances
#     for i, path1 in enumerate(all_paths):
#         for j, path2 in enumerate(all_paths):
#             if i > j: # no need to calculate twice
#                 continue
#             if i == j:
#                 distance_matrix[i, j] = 0
#                 continue
                
#             dataset1 = read_jsonl_dataset(path1)
#             dataset1 = shrink_dataset(dataset1, 100)
#             dataset2 = read_jsonl_dataset(path2)
#             dataset2 = shrink_dataset(dataset2, 100)
#             distance = get_difference_between_datasets(dataset1, dataset2, method=method)
#             distance_matrix[i,j] = distance
#             distance_matrix[j,i] = distance
#             print(path1, path2, distance.item())
    
#     # Second pass: calculate dataset averages for lower triangle
#     num_datasets = len(dataset_paths)
#     dataset_avg_matrix = torch.zeros(num_datasets, num_datasets)
    
#     # Calculate average distances between datasets
#     for dataset_i in range(num_datasets):
#         for dataset_j in range(num_datasets):
#             if dataset_i == dataset_j:
#                 dataset_avg_matrix[dataset_i, dataset_j] = 0
#                 continue
                
#             # Find all AI types for each dataset
#             indices_i = [k for k, d in enumerate(dataset_indices) if d == dataset_i]
#             indices_j = [k for k, d in enumerate(dataset_indices) if d == dataset_j]
            
#             # Calculate average distance between all pairs
#             distances = []
#             for idx_i in indices_i:
#                 for idx_j in indices_j:
#                     # if idx_i == idx_j:
#                     #     continue
#                     distances.append(distance_matrix[idx_i, idx_j].item())
            
#             if len(distances) != 0: # Avoid division by zero in GPT2-GPT2 edge case of only having one AI type in the dataset
#                 dataset_avg_matrix[dataset_i, dataset_j] = sum(distances) / len(distances)
    
#     # Modify distance_matrix: keep upper triangle as is, replace lower triangle with dataset averages
#     for i in range(len(all_paths)):
#         for j in range(len(all_paths)):
#             if i > j:  # Lower triangle
#                 dataset_i = dataset_indices[i]
#                 dataset_j = dataset_indices[j]
#                 distance_matrix[i, j] = dataset_avg_matrix[dataset_i, dataset_j]
    
#     create_distance_heatmap(distance_matrix.numpy(), all_AI_types, location=heatmap_location, numbers=numbers)
#     return

# def compare_datasets_with_eachother(
#         dataset_paths,
#         AI_types_list,
#         version='complete', # versions: complete, dev, test, train
#         heatmap_location='Figures/meta-heatmap.png', # location to save the heatmap
#         method='Wasserstein', # method to use for distance calculation, options: 'Wasserstein', 'Cosine'
#         numbers=True
#         ):
    
#     all_AI_types = []
#     all_paths = []
#     for AI_types in AI_types_list:
#         all_AI_types.extend(AI_types)

#     for (dataset_path, AI_types) in zip(dataset_paths, AI_types_list):
#         for AI_type in AI_types:
#             all_paths.append(f'{dataset_path}/{AI_type}_{version}.jsonl')
    
#     distance_matrix = torch.zeros(len(all_paths), len(all_paths))
    
#     for i, path1 in enumerate(all_paths):
#         for j, path2 in enumerate(all_paths):
#             if i > j: # no need to calculate twice
#                 continue
#             if i == j:
#                 distance_matrix[i, j] = 0 
#                 # if same dataset, distance is 0 (also true when actually running the code. This just reduces the number of calculations)
#                 continue
#             dataset1 = read_jsonl_dataset(path1)
#             dataset1 = shrink_dataset(dataset1, 100)

#             dataset2 = read_jsonl_dataset(path2)
#             dataset2 = shrink_dataset(dataset2, 100)

#             distance = get_difference_between_datasets(dataset1, dataset2, method=method)
#             distance_matrix[i,j] = distance
#             distance_matrix[j,i] = distance
#             print(path1, path2, distance.item())

#     create_distance_heatmap(distance_matrix.numpy(), all_AI_types, location=heatmap_location, numbers=numbers)
#     return

    
if __name__ == "__main__":
    method = 'Cosine' # options: 'Wasserstein', 'Cosine', 'Vocabulary_Overlap', 'Topic_Model', 'TF-IDF'
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

    compare_datasets_with_eachother(
        dataset_paths=[
            'Datasets/Ghostbusters_standardized',
            'Datasets/SemEval_standardized/monolingual',
            'Datasets/SemEval_standardized/multilingual',
            'Datasets/GPT2_standardized'
        ],
        AI_types_list=[
            ['claude', 'gpt', 'gpt_prompt1', 'gpt_prompt2', 'gpt_semantic', 'gpt_writing'],
            ['monolingual_'+ai_type for ai_type in ['bloomz', 'chatGPT', 'cohere', 'complete', 'davinci', 'dolly', 'GPT4']],
            ['multilingual_'+ai_type for ai_type in ['bloomz', 'chatGPT', 'cohere', 'complete', 'davinci', 'dolly', 'GPT4']],
            ['gpt2']
        ],
        version=version,
        heatmap_location=f'Figures/meta_{method}-heatmap-nonumbers-avgd.pdf',
        method=method,
        numbers=False
    )

    pass