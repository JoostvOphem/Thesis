# This script standardizes the datasets for the GHOSTBUSTERS project.
# It takes the raw datasets and converts them into a standard format for further processing.

# The datasets are standardized to have the following structure:
# [id], [text], [label]

# the datasets will be randomised, and split into train/test/val

import os
# import pandas as pd
import json
import random
from dataset_utils import read_jsonl_dataset

random.seed(42)  # For reproducibility
TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT = 0.7, 0.15, 0.15
STANDARDIZE_GHOSTBUSTERS = False
STANDARDIZE_SEMEVAL = False
STANDARDIZE_GPT2 = True

### HELPER FUNCTIONS ###

def find_writing_sessions(dataset_dir):
    paths = [
        os.path.join(dataset_dir, path)
        for path in os.listdir(dataset_dir) 
        if path.endswith('txt')
    ]
    return paths

def find_xlsx_files(dataset_dir):
    paths = [
        os.path.join(dataset_dir, path)
        for path in os.listdir(dataset_dir) 
        if path.endswith('.xlsx')
    ]
    return paths

def create_jsonl_dataset(texts, labels, output_files, dataset_name='', shuffle=True):
    # note: Shuffle is assumed to be true and the current implementation will not quite work if set to false. 
    # The train/val/test split is done after shuffling, and val/test will only have human text if shuffle is false.

    if len(texts) != len(labels):
        raise ValueError("Texts and labels must have the same length")

    if len(output_files) != 4:
        raise ValueError("Output files must be a list of four filenames: train, val, test, and complete")
    
    dataset = list(zip(texts, labels))
    if shuffle:
        random.shuffle(dataset)
    
    ln = len(dataset)
    train_size = int(ln * TRAIN_PERCENT)
    val_size = int(ln * VAL_PERCENT)
    # test_size = ln - train_size - val_size <-- coded in as everything after train + val

    def dump_file(output_file, dataset, start, end):
        with open(output_file, 'w') as f:
            for text, label in dataset[start:end]:
                sample = {
                    "text": text,
                    "label": label
                }
                f.write(json.dumps(sample) + '\n')
    
    for output_file in output_files:
        if output_file.endswith('_train.jsonl'):
            dump_file(
                output_file,
                dataset,
                0, train_size
            )

        elif output_file.endswith('_val.jsonl'):
            dump_file(
                output_file,
                dataset,
                train_size, train_size + val_size
            )
        
        elif output_file.endswith('_test.jsonl'):
            dump_file(
                output_file,
                dataset,
                train_size + val_size, ln
            )
        
        elif output_file.endswith('_complete.jsonl'):
            dump_file(
                output_file,
                dataset,
                0, ln
            )

        else:
            print(f"Output file {output_file} does not match any of the expected patterns. Skipping.")
            continue
        
    print(f"Dataset {dataset_name} saved to {output_files[0], output_files[1], output_files[2]}")
    return

### GHOSTBUSTERS ###
# For ghostbusters this will specifically look at essay/claude for AI text and essay/human for human text

def standardize_ghostbusters_essay(AI_type):
    if AI_type not in ['claude', 'gpt', 'gpt_prompt1', 'gpt_prompt2', 'gpt_semantic', 'gpt_writing']:
        raise ValueError("AI type must be one of: 'claude', 'gpt', 'gpt_prompt1', 'gpt_prompt2', 'gpt_semantic', 'gpt_writing'")

    AI_paths = find_writing_sessions(f'Datasets/GhostBusters/ghostbuster-data-master/essay/{AI_type}')
    human_paths = find_writing_sessions('Datasets/GhostBusters/ghostbuster-data-master/essay/human')

    print("GHOSTBUSTERS dataset: essays")
    print(f"AI paths ({AI_type}): {len(AI_paths)}")
    print(f"Human paths: {len(human_paths)}")
    print()

    texts = []
    labels = []
    for path in AI_paths:
        with open(path, 'r') as f:
            text = f.read()
            text = text.replace('\n\n', '\n') # some datasets (i.e. claude, but not human) have double newlines, to prevent training on that aspect specifically these are removed
            texts.append(text)
            labels.append(1)

    for path in human_paths:
        with open(path, 'r') as f:
            text = f.read()
            text = text.replace('\n\n', '\n') # should not be needed, but just in case
            texts.append(text)
            labels.append(0)

    output_files = [
        f'Datasets/GhostBusters_standardized/{AI_type}_train.jsonl',
        f'Datasets/GhostBusters_standardized/{AI_type}_val.jsonl',
        f'Datasets/GhostBusters_standardized/{AI_type}_test.jsonl',
        f'Datasets/GhostBusters_standardized/{AI_type}_complete.jsonl',
    ]
    create_jsonl_dataset(texts, 
                        labels, 
                        output_files, 
                        dataset_name=f'{AI_type}_human'
                        )

    return


if STANDARDIZE_GHOSTBUSTERS:
    for AI_type in ['claude', 'gpt', 'gpt_prompt1', 'gpt_prompt2', 'gpt_semantic', 'gpt_writing']:
        standardize_ghostbusters_essay(AI_type)
    
    print("---- GHOSTBUSTERS dataset essays standardized ----\n")
else:
    print("---- GHOSTBUSTERS dataset essays NOT standardized ----\n")

### SEMEVAL ###

if STANDARDIZE_SEMEVAL:
    # all on subtask A
    # note: these datasets do not all use the same models
    # train monolingual: chatGPT human davinci dolly cohere 
    # train multilingual: chatGPT human bloomz davinci dolly cohere 
    # val monolingual: human bloomz 
    # val multilingual: human chatGPT davinci 
    # test multilingual: chatGPT human bloomz jais-30b davinci dolly cohere llama2-fine-tuned 
    # test multilingual: chatGPT human bloomz jais-30b davinci dolly cohere llama2-fine-tuned

    val_monolingual = read_jsonl_dataset('Datasets/SemEval/SemEval2024-M4/SubtaskA/subtaskA_dev_monolingual.jsonl', additional_field_names=['model'])
    val_multilingual = read_jsonl_dataset('Datasets/SemEval/SemEval2024-M4/SubtaskA/subtaskA_dev_multilingual.jsonl', additional_field_names=['model'])

    test_monolingual = read_jsonl_dataset('Datasets/SemEval/SemEval2024-M4/SubtaskA/subtaskA_test_monolingual.jsonl', additional_field_names=['model'])
    test_multilingual = read_jsonl_dataset('Datasets/SemEval/SemEval2024-M4/SubtaskA/subtaskA_test_multilingual.jsonl', additional_field_names=['model'])

    train_monolingual = read_jsonl_dataset('Datasets/SemEval/SemEval2024-M4/SubtaskA/subtaskA_train_monolingual.jsonl', additional_field_names=['model'])
    train_multilingual = read_jsonl_dataset('Datasets/SemEval/SemEval2024-M4/SubtaskA/subtaskA_train_multilingual.jsonl', additional_field_names=['model'])

    def make_datasets(val, test, train, typ):
        model_types = []
        for model in set(train['model']):
            model_types.append(model)
        for model in set(val_monolingual['model']):
            model_types.append(model)
        for model in set(test_monolingual['model']):
            model_types.append(model)
        model_types = list(set(model_types))

        # pull together all texts and labels from the monolingual datasets
        all_texts = []
        all_labels = []

        human_texts = []
        human_labels = []
        for i in range(len(train['text'])):
            if train['model'][i] == 'human':
                human_texts.append(train['text'][i])
                human_labels.append(train['label'][i])
        for i in range(len(val['text'])):
            if val['model'][i] == 'human':
                human_texts.append(val['text'][i])
                human_labels.append(val['label'][i])
        for i in range(len(test['text'])):
            if test['model'][i] == 'human':
                human_texts.append(test['text'][i])
                human_labels.append(test['label'][i])
        all_texts.extend(human_texts)
        all_labels.extend(human_labels)

        AI_texts = []
        AI_labels = []
        for model in model_types:
            if model == 'human':
                continue

            for i in range(len(train_monolingual['text'])):
                if train_monolingual['model'][i] == model:
                    AI_texts.append(train_monolingual['text'][i])
                    AI_labels.append(train_monolingual['label'][i])
            for i in range(len(val_monolingual['text'])):
                if val_monolingual['model'][i] == model:
                    AI_texts.append(val_monolingual['text'][i])
                    AI_labels.append(val_monolingual['label'][i])
            for i in range(len(test_monolingual['text'])):
                if test_monolingual['model'][i] == model:
                    AI_texts.append(test_monolingual['text'][i])
                    AI_labels.append(test_monolingual['label'][i])
            
            texts = human_texts + AI_texts
            labels = human_labels + AI_labels

            create_jsonl_dataset(
                texts, 
                labels, 
                [
                    f'Datasets/SemEval_standardized/{typ}/{typ}_{model}_train.jsonl',
                    f'Datasets/SemEval_standardized/{typ}/{typ}_{model}_val.jsonl',
                    f'Datasets/SemEval_standardized/{typ}/{typ}_{model}_test.jsonl',
                    f'Datasets/SemEval_standardized/{typ}/{typ}_{model}_complete.jsonl'
                ],
                dataset_name=f'{typ}__{model}'
            )

            all_texts.extend(AI_texts)
            all_labels.extend(AI_labels)
        
        create_jsonl_dataset(
            all_texts, 
            all_labels, 
            [
                f'Datasets/SemEval_standardized/{typ}/{typ}_complete_train.jsonl',
                f'Datasets/SemEval_standardized/{typ}/{typ}_complete_val.jsonl',
                f'Datasets/SemEval_standardized/{typ}/{typ}_complete_test.jsonl',
                f'Datasets/SemEval_standardized/{typ}/{typ}_complete_complete.jsonl'
            ],
            dataset_name=f'{typ}_complete'
        )
    make_datasets(val_monolingual, test_monolingual, train_monolingual, typ='monolingual')
    make_datasets(val_multilingual, test_multilingual, train_multilingual, typ='multilingual')

    print("---- SEMEVAL dataset essays standardized ----\n")

else:
    print("---- SEMEVAL dataset essays NOT standardized ----\n")


### GPT2 ###

if STANDARDIZE_GPT2:
    import pandas as pd

    files = find_xlsx_files('Datasets/GPT2')
    all_AI_texts = []
    all_human_texts = []
    for i, file in enumerate(files):

        sheets_dict = pd.read_excel(file, sheet_name=None)

        all_sheets = []
        for name, sheet in sheets_dict.items():
            sheet['sheet'] = name
            sheet = sheet.rename(columns=lambda x: x.split('\n')[-1])
            all_sheets.append(sheet)

        full_table = pd.concat(all_sheets)
        full_table.reset_index(inplace=True, drop=True)

        
        if not full_table.columns[-2] == 'AI': # remove the 'AI or Human?' columns
            continue
                        
        # make sure no human and AI texts of the same questions are put in the preprocessed dataset
        if i % 2 == 0:
            all_AI_texts.extend(full_table['AI'].tolist())
        else:
            all_human_texts.extend(full_table['Human'].tolist())

    # remove the nan-values
    all_AI_texts = [text for text in all_AI_texts if isinstance(text, str)]
    all_human_texts = [text for text in all_human_texts if isinstance(text, str)]
   
    texts = all_AI_texts + all_human_texts
    labels = [1] * len(all_AI_texts) + [0] * len(all_human_texts)

    create_jsonl_dataset(
        texts, 
        labels, 
        [
            'Datasets/GPT2_standardized/gpt2_train.jsonl',
            'Datasets/GPT2_standardized/gpt2_val.jsonl',
            'Datasets/GPT2_standardized/gpt2_test.jsonl',
            'Datasets/GPT2_standardized/gpt2_complete.jsonl'
        ],
        dataset_name='gpt2'
    )
    print("---- GPT2 dataset standardized ----\n")

else:
    print("---- GPT2 dataset NOT standardized ----\n")
