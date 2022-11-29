# Обучите модель логистической регрессии на эмбеддингах.

# Напечатайте на экране значение accuracy на обучающей выборке.
# Чтобы не создавать эмбеддинги слишком долго, возьмите из выборки
# только 400 случайных элементов. Для корректного тестирования
# поделите их на обучающую и тестовую выборки в соотношении 50:50.
# Целевой признак находится в переменной df_tweets['positive'].

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import notebook
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

df_tweets = pd.read_csv('/datasets/tweets.csv')

tokenizer = transformers.BertTokenizer(
    vocab_file='/datasets/ds_bert/vocab.txt')

tokenized = df_tweets['text'].apply(
    lambda x: tokenizer.encode(x, add_special_tokens=True))

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len - len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)

config = transformers.BertConfig.from_json_file(
    '/datasets/ds_bert/bert_config.json')
model = transformers.BertModel.from_pretrained(
    '/datasets/ds_bert/rubert_model.bin', config=config)