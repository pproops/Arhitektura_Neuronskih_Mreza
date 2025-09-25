import os
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("imdb")

os.makedirs("data", exist_ok=True)
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])
train_df.to_csv("data/imdb_train.csv", index=False)
test_df.to_csv("data/imdb_test.csv", index=False)
