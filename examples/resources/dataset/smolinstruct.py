import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import random
from tqdm import tqdm

def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    
    tokenizer.sep_token = '<unk>'
    tokenizer.cls_token = '<unk>'
    tokenizer.mask_token = '<unk>'
    
    tokenizer.pad_token_id = 0  # unk token
    tokenizer.padding_side = "left"  # Important for generation
    
    return tokenizer

def generate_chat(input_text: str, output_text: str = None, prefix_chat=None):
    chat = [
        {"role": "user", "content": input_text},
    ]
    if output_text is not None:
        chat.append({"role": "assistant", "content": output_text})
    if prefix_chat is not None:
        chat = prefix_chat + chat
    return chat

def generate_prompt(chat):
    if len(chat) == 1:
        return f"<s>[INST] {chat[0]['content']} [/INST]"
    else:
        return f"<s>[INST] {chat[0]['content']} [/INST] {chat[1]['content']}</s>"

def tokenize_prompt(tokenizer, prompt, cutoff_len=512, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point, tokenizer, cutoff_len=512, train_on_inputs=False, add_eos_token=False):
    # Handle different data formats
    if 'instruction' in data_point and 'response' in data_point:
        input_text = data_point['instruction']
        output_text = data_point['response']
    elif 'input' in data_point and 'output' in data_point:
        input_text = data_point['input']
        output_text = data_point['output']
    else:
        raise ValueError(f"Unsupported data format. Keys: {data_point.keys()}")
        
    # Generate chat and full prompt
    chat = generate_chat(input_text, output_text)
    full_prompt = generate_prompt(chat)
    tokenized_full_prompt = tokenize_prompt(tokenizer, full_prompt, cutoff_len, add_eos_token)

    # Mask inputs if needed (don't train on user prompt)
    if not train_on_inputs:
        user_chat = generate_chat(input_text, output_text=None)
        user_prompt = generate_prompt(user_chat)
        tokenized_user_prompt = tokenize_prompt(tokenizer, user_prompt, cutoff_len, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        
        # Mask the user prompt part
        tokenized_full_prompt["labels"] = (
            [-100] * user_prompt_len + 
            tokenized_full_prompt["labels"][user_prompt_len:]
        )

    return tokenized_full_prompt

def get_smolinstruct_data(model_name, tasks, nsamples, seqlen, train_on_inputs=False, add_eos_token=False):
    tokenizer = get_tokenizer(model_name)
    
    if tasks:
        train_dataset = load_dataset("osunlp/SMolInstruct", split="train", tasks=tasks)
    else:
        train_dataset = load_dataset("osunlp/SMolInstruct", split="train")
    
    try:
        if tasks:
            val_dataset = load_dataset("osunlp/SMolInstruct", split="validation", tasks=tasks)
        else:
            val_dataset = load_dataset("osunlp/SMolInstruct", split="validation")
    except:
        print("No validation split found, will create from train split")
        val_dataset = None

    # Limit samples
    if nsamples and nsamples < len(train_dataset):
        train_dataset = train_dataset.shuffle().select(range(nsamples))

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in train_dataset:
        try:
            tokenized = generate_and_tokenize_prompt(item, tokenizer, seqlen, train_on_inputs, add_eos_token)
            
            input_ids_list.append(torch.tensor(tokenized['input_ids']))
            attention_mask_list.append(torch.tensor(tokenized['attention_mask']))
            labels_list.append(torch.tensor(tokenized['labels']))
        except Exception as e:
            continue
    
    batch_items = list(zip(input_ids_list, attention_mask_list, labels_list))

    class ValidationWrapper:
        def __init__(self, dataset):
            self.dataset = dataset
    
    if val_dataset is not None:
        valenc = ValidationWrapper(val_dataset.select(range(min(100, len(val_dataset)))))
    else:
        valenc = ValidationWrapper(None)

    return batch_items, valenc
