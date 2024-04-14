import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
import torch

os.environ["WANDB_DISABLED"] = "true"

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            print(f"Epoch: {state.epoch}, Step: {state.global_step}, Loss: {logs['loss']}")

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
    df = pd.DataFrame(lyrics, columns=['lyrics'])

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    dataset_lyrics = Dataset.from_pandas(df)
    tokenized_text = dataset_lyrics.map(tokenizer_fn, batched=True)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    training_args = TrainingArguments(
        output_dir='./fine-tuned_model',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1, 
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
    model.save_pretrained('./fine-tuned_model')
    tokenizer.save_pretrained('./fine-tuned_model')


if __name__ == "__main__":
    fine_tune()
