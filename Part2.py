import pandas as pd
import numpy as np


def pprint(smth):
    print(smth)
    print()


# 1) Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
df = pd.read_csv('titanic/train.csv', index_col='PassengerId')


