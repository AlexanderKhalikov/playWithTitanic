import pandas as pd
import numpy as np


def pprint(smth):
    print(smth)
    print()


# 1) Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
df = pd.read_csv('titanic/train.csv', index_col='PassengerId')


# 2) Оставьте в выборке четыре признака: класс пассажира (Pclass),
# цену билета (Fare), возраст пассажира (Age) и его пол (Sex).

x_labels = ['Pclass', 'Sex', 'Age', 'Fare']
X = df.loc[:, x_labels]

pprint(X)

# 3) Обратите внимание, что признак Sex имеет строковые значения. Замените их на числовые


