import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import torch

def test():
    # to load
    model = GPT2LMHeadModel.from_pretrained('./fine-tuned_model')
    tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned_model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt = "Write the lyrics to a love song."
    input_tok = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
    tokenizer.pad_token = tokenizer.eos_token

    result = model.generate(input_ids=input_tok["input_ids"], max_length=30, do_sample=True, top_p=0.9, temperature=0.7, pad_token_id=tokenizer.pad_token_id)

    result_text = tokenizer.decode(result[0], skip_special_tokens=True)
    return result_text

if __name__ == "__main__":
    lyrics = test()
    print(lyrics)
