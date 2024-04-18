import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import torch

def test():
    # to load fine-tuned model
    model = GPT2LMHeadModel.from_pretrained('./fine-tuned_model_new')
    tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned_model_new')

    # pre-trained model
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = GPT2LMHeadModel.from_pretrained('gpt2')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #prompt = "Write a song lyrics that starts with the following text: I love you more everyday because"
    #prompt = "Write a song lyrics that starts with the following text: It's over now because we couldn't"
    #prompt = "Write a song lyrics that starts with the following text: I've been so sad ever since"
    #prompt = "Write a song lyrics that starts with the following text: I've been so happy ever since"
    #prompt = "Write a song lyrics that starts with the following text: Once upon a time we were"
    #prompt = "Write a song lyrics that starts with the following text: nicki minaj yo now we gonna"
    #prompt = "Write a song lyrics that starts with the following text: I want to go somewhere far"
    #prompt = "Write a song lyrics that starts with the following text: girl this song is for you"
    #prompt = "Write a song lyrics that starts with the following text: I want to change the world"
    prompt = "Write a song lyrics that starts with the following text: I can't wait for summer to"
    input_tok = tokenizer(prompt, return_tensors='pt').to(device)
    
    result = model.generate(input_ids=input_tok["input_ids"], attention_mask=input_tok["attention_mask"], max_length=145, do_sample=True, top_p=0.8, temperature=0.7, repetition_penalty=1.2)

    result_text = tokenizer.decode(result[0], skip_special_tokens=True)
    return result_text

if __name__ == "__main__":
    lyrics = test()
    print(lyrics)
