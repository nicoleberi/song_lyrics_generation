import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
import torch
import random
import matplotlib.pyplot as plt

os.environ["WANDB_DISABLED"] = "true"
epoch_vals = []
loss_vals = []

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            print(f"Epoch: {state.epoch}, Step: {state.global_step}, Loss: {logs['loss']}")
            epoch_vals.append(state.epoch)
            loss_vals.append(logs['loss'])

def fine_tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenizer_fn(batch):
        t = tokenizer(batch['lyrics'], truncation=True, padding="max_length", max_length=1024, return_tensors="pt")
        return {key: value.to(device) for key, value in t.items()}
    
    lyrics = []
    for f in os.listdir('csv'):
        df = pd.read_csv('csv/' + f)
        for song in df['Lyric'].dropna():
            lyrics.append(song)
    random.shuffle(lyrics)
    df = pd.DataFrame(lyrics, columns=['lyrics'])

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    dataset_lyrics = Dataset.from_pandas(df)
    tokenized_text = dataset_lyrics.map(tokenizer_fn, batched=True)

    training_args = TrainingArguments(
        output_dir='./fine-tuned_model_new',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        logging_dir='./logs_new',
        logging_steps=100, 
        save_strategy="epoch",
        overwrite_output_dir=True
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print_loss_callback = PrintLossCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_text,
        callbacks=[print_loss_callback]
    )

    trainer.train()
    model.save_pretrained('./fine-tuned_model_new')
    tokenizer.save_pretrained('./fine-tuned_model_new')


if __name__ == "__main__":
    fine_tune()
    fig, ax = plt.subplots()
    ax.plot(epoch_vals, loss_vals)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss value') 
    fig.savefig('loss.png')
