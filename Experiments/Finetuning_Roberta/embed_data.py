import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class Classifier(nn.Module):
    def __init__(self, hidden_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings):
        return self.sigmoid(self.linear(embeddings))
    
MODEL_NAME = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
roberta = RobertaModel.from_pretrained(MODEL_NAME)

# Freeze it (not strictly necessary for inference but good practice)
for param in roberta.parameters():
    param.requires_grad = False

classifier = Classifier(hidden_dim=768)

state_dict = torch.load("best_classifier.pt")
remove_prefix = '0'
state_dict = {'linear'+k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}

classifier.load_state_dict(state_dict)
classifier.eval()

def classify_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = roberta(**inputs)
        pooled = outputs.pooler_output  # shape: [1, 768]
        prediction = classifier(pooled)
    return prediction.item()

def embed_layer(text, layer=-2): # second to last layer, kinda arbitrarily chosen for now
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = roberta(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        cls_embedding = hidden_states[layer][:, 0, :]  # use the [CLS] token
    return cls_embedding

text = "This is a sample input to test the classifier."
score = classify_text(text)

print(f"Prediction score: {score:.4f}")
print(f"embedding: {embed_layer(text)}")
print("Label:", "Human" if score >= 0.5 else "AI")