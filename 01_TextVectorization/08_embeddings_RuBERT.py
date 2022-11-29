# Эмбеддинги RuBERT
# Почти всё готово, чтобы сформировать векторы моделью RuBERT и классифицировать твиты. В дело вступают тензоры.
# Инициализируем конфигурацию BertConfig (англ. Bert Configuration). В качестве аргумента передадим ей JSON-файл с описанием настроек модели. JSON (англ. JavaScript Object Notation, «объектная запись JavaScript») — это организованный по ключам поток цифр, букв, двоеточий и фигурных скобок, который возвращает сервер при запросе.
# Затем инициализируем саму модель класса BertModel. Передадим ей файл с предобученной моделью и конфигурацией:

config = transformers.BertConfig.from_json_file(
    '/datasets/ds_bert/bert_config.json')
model = transformers.BertModel.from_pretrained(
    '/datasets/ds_bert/rubert_model.bin', config=config)

# Начнём преобразование текстов в эмбеддинги. Это может занять
# несколько минут, поэтому подключим библиотеку tqdm (араб.
# taqadum, تقدّم, «прогресс»). Она нужна, чтобы наглядно показать
# индикатор прогресса. В Jupyter применим функцию notebook() из
# этой библиотеки:

from tqdm import notebook
import pandas as pd

# Эмбеддинги модель BERT создаёт батчами. Чтобы хватило оперативной памяти, сделаем размер батча небольшим:
batch_size = 100

# Сделаем цикл по батчам. Отображать прогресс будет функция notebook():
# сделаем пустой список для хранения эмбеддингов твитов
embeddings = []

for i in notebook.tqdm(range(padded.shape[0] // batch_size)):
    ...
# Преобразуем данные в формат тензоров (англ. tensor) —
# многомерных векторов в библиотеке torch. Тип данных LongTensor
# (англ. «длинный тензор») хранит числа в «длинном формате», то
# есть выделяет на каждое число 64 бита.

# преобразуем данные
batch = torch.LongTensor(padded[batch_size*i:batch_size*(i+1)])
# преобразуем маску
attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)])

# Чтобы получить эмбеддинги для батча, передадим модели данные и маску:
batch_embeddings = model(batch, attention_mask=attention_mask_batch)

# Для ускорения вычисления функцией no_grad() (англ. no gradient,
# «нет градиента») в библиотеке torch укажем, что градиенты не
# нужны: модель BERT обучать не будем.
with torch.no_grad():
    batch_embeddings = model(batch, attention_mask=attention_mask_batch)

# Из полученного тензора извлечём нужные элементы и добавим в список всех эмбеддингов:
# преобразуем элементы методом numpy() к типу numpy.array
embeddings.append(batch_embeddings[0][:,0,:].numpy())

embeddings = []

for i in notebook.tqdm(range(padded.shape[0] // batch_size)):
    batch = torch.LongTensor(padded[batch_size*i:batch_size*(i+1)])
    attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)])

    with torch.no_grad():
        batch_embeddings = model(batch, attention_mask=attention_mask_batch)

    embeddings.append(batch_embeddings[0][:,0,:].numpy())

# Соберём все эмбеддинги в матрицу признаков вызовом функции
# concatenate():
features = np.concatenate(embeddings)
# Признаки готовы, можно обучать модель!