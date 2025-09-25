import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

full_test_df = pd.read_csv('data/imdb_test.csv')

positive_samples = full_test_df[full_test_df['label'] == 1].head(50)
negative_samples = full_test_df[full_test_df['label'] == 0].head(50)
test_df = pd.concat([positive_samples, negative_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

import os
model_dir = './results'
if os.path.exists(model_dir) and any(os.scandir(model_dir)):
    model = BertForSequenceClassification.from_pretrained(model_dir)
else:
    print("Nije pronađen lokalni trenirani model, koristi se originalni BERT.")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

batch_size = 32
preds = []
for i in range(0, len(test_texts), batch_size):
    batch_texts = test_texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        batch_preds = torch.argmax(outputs.logits, dim=1).numpy()
        preds.extend(batch_preds)

cm = confusion_matrix(test_labels, preds, labels=[0, 1])
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap='Blues')
plt.title('Matrica konfuzije')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.xticks([0,1], ['Negativno', 'Pozitivno'])
plt.yticks([0,1], ['Negativno', 'Pozitivno'])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

plt.figure(figsize=(6,4))
plt.hist(preds, bins=2, rwidth=0.8)
plt.title('Distribucija predikcija sentimenta')
plt.xlabel('Sentiment (0=Negativno, 1=Pozitivno)')
plt.ylabel('Broj primjera')
plt.savefig('sentiment_distribution.png')
plt.show()

plt.figure(figsize=(6,4))
plt.hist(test_labels, bins=2, rwidth=0.8)
plt.title('Distribucija stvarnih labela u test skupu')
plt.xlabel('Sentiment (0=Negativno, 1=Pozitivno)')
plt.ylabel('Broj primjera')
plt.savefig('label_distribution.png')
plt.show()

classification_report(test_labels, preds, target_names=['Negativno', 'Pozitivno'], labels=[0, 1])
