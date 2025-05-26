from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np

# Take a sample of embeddings
train_data = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_train.jsonl")))
train_labels = torch.tensor(np.array(torch.load("Datasets/Ghostbusters_standardized_embedded/gpt_writing_train_labels.pt")))


sample_data = train_data.numpy()
sample_labels = train_labels.numpy()

# Reduce dimensions for visualization
tsne = TSNE(n_components=2, random_state=42)

embeddings_2d = tsne.fit_transform(sample_data.reshape(-1, sample_data.shape[-1]))

# Plot
plt.figure(figsize=(10, 8))
colors = ['red', 'blue']
for i in range(2):
    mask = sample_labels == i
    mask = mask.flatten()
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                c=colors[i], label=f'Class {i}', alpha=0.6)
plt.legend()
plt.title('Embedding Space Visualization (Ghostbusters GPT Writing)')
plt.show()