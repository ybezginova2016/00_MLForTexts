# Лемматизация

# Прежде чем извлечь признаки из текста, упростим его.
# Рассмотрим этапы предобработки текста:

# Токенизация (англ. tokenization) — разбиение текста на токены:
# отдельные фразы, слова, символы.
# Лемматизация (англ. lemmatization) — приведение слова к начальной форме —
# лемме.
# Функция лемматизации русского текста есть в библиотеках:

# pymorphy2 (англ. python morphology, «морфология для Python»), **
# UDPipe (англ. universal dependencies pipeline, «конвейер для построения
# общих зависимостей»), pymystem3.

# Рассмотрим, как работать с библиотекой pymystem3.
# Создадим класс для лемматизации:

from pymystem3 import Mystem
m = Mystem()

# Функции lemmatize() передадим «лемматизируй это»:
print(m.lemmatize("лемматизируй это"))

# Функция вернула лемму каждого слова. Для функции конец строки
# — это тоже знак, поэтому находится и его лемма (\n)
# (от англ. new line, «новая строка»).

# Функцией join() объединим элементы списка в строку,
# разделив их пробелом (но можно и без него):
print(" ".join(['лемматизировать', ' ', 'это', '\n']))

# RuTweetCorp https://study.mokoron.com/

# Перед вами уменьшенный датасет — 5000 записей. Каждая запись
# содержит текст поста и оценку его тональности. Если пост
# позитивный, то метка «1», если негативный — «0». Осторожно:
# твиты могут содержать обсценную (от лат. obscenus,
# «непристойный») лексику.

#### EXERCISE 1 ####
# Напишите функцию lemmatize(text). На вход она принимает текст из
# корпуса и возвращает лемматизированную строку. Возьмите первый
# текст из датасета tweets.csv. Код напечатает на экране исходный
# и лематизированный тексты.

import pandas as pd
data = pd.read_csv('tweets.csv')
# Изменим кодировку методом astype():
corpus = data['text'].values.astype('U')

import pandas as pd
from pymystem3 import Mystem

data = pd.read_csv('/datasets/tweets.csv')

# Переведём тексты в стандартный для Python формат: кодировку Unicode U.
# Изменим кодировку методом astype():
corpus = data['text'].values.astype('U')


def lemmatize(text):
    m = Mystem()
    # объединим элементы в одну строку пробелом или без него
    lem = ''.join(m.lemmatize(text))
    return lem

print("Исходный текст:", corpus[0])
print("Лемматизированный текст:", lemmatize(corpus[0]))