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

X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})
pprint(X['Sex'])
# 4) Выделите целевую переменную — она записана в столбце Survived.

y = df['Survived']
pprint(y)

# 5) В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст.
# Такие записи при чтении их в pandas принимают значение nan.
# Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.

X = X.dropna(axis=0)
y = y[X.index.values]

pprint(X.info())

# 6) Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию.

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=241)
tree.fit(np.array(X.values), np.array(y.values))

# 7) Вычислите важности признаков и найдите два признака с наиболь- шей важностью.
# Их названия будут ответами для данной задачи (в качестве ответа укажите названия признаков через запятую без пробелов)

importances = pd.Series(tree.feature_importances_, index=x_labels)
pprint(importances.sort_values(ascending=False))
