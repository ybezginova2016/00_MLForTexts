# Мешок слов и N-граммы

# Переведём тексты в понятный для машины формат — векторный.
# Преобразовать слова в векторы поможет модель «мешок слов»
# (англ. bag of words). Она преобразует текст в вектор, не
# учитывая порядок слов. Отсюда и название — «мешок».

# Создание мешка слов
# Научимся создавать мешок слов и находить стоп-слова (англ.
# stopwords).
# Чтобы преобразовать корпус текстов в мешок слов, обратимся
# к классу CountVectorizer() (англ. count vectorizer,
# «счётчик слов для создания векторов»). Он находится в модуле
# sklearn.feature_extraction.text (англ. «построение признаков
# текста»).

# Импортируем его:

from sklearn.feature_extraction.text import CountVectorizer
# создадим счетчик
count_vect = CountVectorizer()

# Передадим счётчику корпус текстов. Для этого вызовем знакомую
# вам функцию fit_transform(). Счётчик выделит из корпуса уникальные
# слова и посчитает количество их вхождений в каждом тексте корпуса.
# Отдельные буквы счётчик как слова не учитывает.

# bow, от англ. bag of words
bow = count_vect.fit_transform(corpus)

#### EXERCISE 1 ####
# 1. В файле tweets_lemm.csv находится датасет, в который мы
# добавили столбец lemm_text с очищенными и лемматизированными
# текстами.
# Создайте два мешка слов для корпуса твитов: очищенный от стоп-слов
# и с ними. Напечатайте их размеры на экране.
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("/datasets/tweets_lemm.csv")
corpus = data['lemm_text'].values.astype('U')

# Список стоп-слов указывается при вызове CountVectorizer().
# Создадим счётчик:
count_vect = CountVectorizer()

# Передадим счётчику корпус текстов. Для этого вызовем знакомую вам функцию fit_transform(). Счётчик выделит из корпуса уникальные слова и посчитает количество их вхождений в каждом тексте корпуса. Отдельные буквы счётчик как слова не учитывает.

# создайте мешок слов без учёта стоп-слов
# bow, от англ. bag of words
bow = count_vect.fit_transform(corpus)

print("Размер мешка без учёта стоп-слов:", bow.shape)

from nltk.corpus import stopwords
# Чтобы пакет заработал, загрузим список стоп-слов:
import nltk
nltk.download('stopwords')

# создайте новый мешок слов с учётом стоп-слов
stop_words = set(stopwords.words('russian'))

count_vect_nostops = CountVectorizer(stop_words=stop_words)
bow = count_vect_nostops.fit_transform(corpus)

print("Размер мешка с учётом стоп-слов:", bow.shape)

# Output
# Размер мешка без учёта стоп-слов: (5000, 9345)
# Размер мешка с учётом стоп-слов: (5000, 9248)

#### EXERCISE 2 ####
# 2. Создайте счётчик N-грамм для корпуса твитов. В каждой фразе
# по два слова. Напечатайте на экране размер N-грамм.

# Размер N-грамм указывается при вызове CountVectorizer().

# CountVectorizer() также нужен для расчёта N-грамм. Чтобы он
# считал словосочетания, укажем размер N-граммы через аргумент
# ngram_range (англ. «диапазон N-грамм»). Например, если мы ищем
# словосочетания по два слова в фразе, то диапазон зададим такой:
# count_vect = CountVectorizer(ngram_range=(2, 2))

import pandas as pd
# импортируйте CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("/datasets/tweets_lemm.csv")
corpus = list(data['lemm_text'])

# создайте n-грамму n_gramm, для которой n=2
count_vect = CountVectorizer(ngram_range=(2, 2))

n_gramm = count_vect.fit_transform(corpus)

print("Размер:", n_gramm.shape)

# Output
# Размер: (5000, 32701)
# Словосочетаний больше, чем уникальных слов
