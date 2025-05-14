import json
import os
import random

def read_jsonl_dataset(input_file, additional_field_names=None):
    """
    Read a JSONL file back into lists of texts and labels.
    
    Args:
    input_file (str): Path to the input JSONL file
    
    Returns:
    tuple: Lists of texts and labels
    """
    texts = []
    labels = []

    output = {}
    if not additional_field_names is None:
        add_fields = {name: [] for name in additional_field_names}

    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            texts.append(sample['text'])
            labels.append(sample['label'])

            if not additional_field_names is None:
                for name in additional_field_names:
                    add_fields[name].append(sample[name])


    output['text'] = texts
    output['label'] = labels
    if not additional_field_names is None:
        for name in additional_field_names:
            output[name] = add_fields[name]
    
    return output