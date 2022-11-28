#### TF-IDF ####

# Мешок слов учитывает частоту употребления слов. Посмотрим, как
# часто уникальное слово встречается во всём корпусе и в отдельном
# его тексте.

# Оценка важности слова определяется величиной TF-IDF (от англ.
# term frequency, «частота терма, или слова»; inverse document
# frequency, «обратная частота документа, или текста»). То есть TF
# отвечает за количество упоминаний слова в отдельном тексте, а
# IDF отражает частоту его употребления во всём корпусе.

# TFIDF = TF * IDF

# TF рассчитывается так: TF = t / n,

# где t (от англ. term) — количество употребления слова, а n —
# общее число слов в тексте.

# IDF нужна в формуле, чтобы уменьшить вес слов, наиболее
# распространённых в любом другом тексте заданного корпуса.
# IDF зависит от общего числа текстов в корпусе (D) и количества
# текстов, в которых это слово встречается (d).

# Большая величина TF-IDF говорит об уникальности слова в тексте
# по отношению к корпусу. Чем чаще оно встречается в конкретном
# тексте и реже в остальных, тем выше значение TF-IDF.

# Рассчитать TF-IDF можно и в библиотеке sklearn. Класс TfidfVectorizer()
# (от англ. «счётчик величин TF-IDF») находится в модуле
# sklearn.feature_extraction.text. Импортируем его:

from sklearn.feature_extraction.text import TfidfVectorizer
# По аналогии с CountVectorizer() создадим счётчик, указав в нём
# стоп-слова:

count_tf_idf = TfidfVectorizer(stop_words=stopwords)

# Чтобы посчитать TF-IDF для корпуса текстов,
# вызовем функцию fit_transform():
tf_idf = count_tf_idf.fit_transform(corpus)

# Передав TfidfVectorizer() аргумент ngram_range, можно рассчитать
# N-граммы.
# Если данные разделены на обучающую и тестовую выборки,
# функцию fit() запускайте только на обучающей. Иначе тестирование
# будет нечестным: в модели будут учтены частоты слов из тестовой
# выборки.

#### EXERCISE ####

# Создайте матрицу cо значениями TF-IDF по корпусу твитов.
# Сохраните её в переменной tf_idf. Укажите стоп-слова. Напечатайте
# на экране размер матрицы.

# Список стоп-слов указывается при вызове TfidfVectorizer().

import pandas as pd
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("/datasets/tweets_lemm.csv")
corpus = data['lemm_text'].values.astype('U')

nltk.download('stopwords')
stopwords = set(nltk_stopwords.words('russian'))

# По аналогии с CountVectorizer() создадим счётчик, указав в нём стоп-слова:
count_tf_idf = TfidfVectorizer(stop_words=stopwords)

tf_idf = count_tf_idf.fit_transform(corpus)

print("Размер матрицы:", tf_idf.shape)

# Output
# Размер матрицы: (5000, 9248)


