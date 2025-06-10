from dataset_utils import get_roberta_and_tokenizer, embed_layer, get_input_output_files
import pandas as pd
import torch

DATA_TO_EMBED = "gpt2"
ROBERTA_TO_USE = "ghostbusters_ALL"
AMT_TO_EMBED = 5000


input_files, output_folder = get_input_output_files(DATA_TO_EMBED, 
                                                    other_output_folder=ROBERTA_TO_USE)

model, tokenizer = get_roberta_and_tokenizer(ROBERTA_TO_USE)

for input_file in input_files:
    print(input_file)
    jsonObj = pd.read_json(path_or_buf=input_file, lines=True)
    jsonObj = jsonObj.reset_index(drop=True)

    embeddings = []
    labels = []

    for i, row in jsonObj.iterrows():
        if i % 100 == 0 and i > 0:
            print(f"Processing row {i} / {len(jsonObj)} of {input_file}")
        
        if i == AMT_TO_EMBED:
            break
        text = row['text']
        embedding = embed_layer(
            model,
            tokenizer,
            text,
            layer=-2)
        embeddings.append(embedding)

        label = row['label']
        labels.append(label)
        
    final_text_tensor = torch.stack(embeddings)
    final_label_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    file_ending = input_file.split("/")[-1]
    torch.save(final_text_tensor, output_folder + "/" + file_ending)
    torch.save(final_label_tensor, output_folder + "/" + file_ending.replace('.jsonl', '_labels.pt'))
    print(f"Saved {file_ending} to {output_folder}")