import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset


full_train_df = pd.read_csv('data/imdb_train.csv')
full_test_df = pd.read_csv('data/imdb_test.csv')

train_pos = full_train_df[full_train_df['label'] == 1].head(500)
train_neg = full_train_df[full_train_df['label'] == 0].head(500)
train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

test_pos = full_test_df[full_test_df['label'] == 1].head(500)
test_neg = full_test_df[full_test_df['label'] == 0].head(500)
test_df = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42).reset_index(drop=True)



train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()
test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels}).map(tokenize_function, batched=True)
test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels}).map(tokenize_function, batched=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

results = trainer.evaluate()

model.save_pretrained('./results')
tokenizer.save_pretrained('./results')
