# Предобработка твитов
import numpy as np
import torch
import transformers

# инициализируем токенизатор
tokenizer = transformers.BertTokenizer(
    vocab_file='/datasets/ds_bert/vocab.txt')

# токенизируем текст
vector = tokenizer.encode('Очень удобно использовать уже готовый трансформатор текста', add_special_tokens=True)

# применим padding к векторам
n = 280
# англ. вектор с отступами
padded = vector + [0]*(n - len(vector))

# создадим маску для важных токенов
attention_mask = np.where(padded != 0, 1, 0)

# Адаптируйте код предобработки из уроков к задаче классификации твитов:
# Загрузите файл с твитами /datasets/tweets.csv.
# Токенизируйте каждый твит.
# Найдите максимальную длину векторов после токенизации.
# Примените padding к векторам и создайте маску для
# выделения важных токенов.

import numpy as np
import torch
import transformers
import pandas as pd

all_tweets = pd.read_csv('/datasets/tweets.csv')
# print(all_tweets)

# инициализируем токенизатор
tokenizer = transformers.BertTokenizer(
    vocab_file='/datasets/tweets.csv')

# токенизируем все твиты
tokenized = all_tweets['text'].apply(
  lambda x: tokenizer.encode(x, add_special_tokens=True))

# применим padding к векторам
n = 280
# англ. вектор с отступами
padded = tokenized + [0]*(n - len(tokenized))

# создадим маску для важных токенов
attention_mask = np.where(padded != 0, 1, 0)