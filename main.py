import pandas as pd
import numpy as np
import sklearn as sk
from pandas_profiling import ProfileReport

data = pd.read_csv('titanic/train.csv')
# ProfileReport(data).to_file('report.html')

# 1. Какое количество мужчин и женщин ехало на корабле?
# В качестве ответа приведите два числа через пробел.

# print(data['Sex'].value_counts())

# 2. Какой части пассажиров удалось выжить?
# Посчитайте долю выживших пассажиров. Ответ приведите в процентах
# (число в интервале от 0 до 100, знак процента не нужен).
surviced_counts = data['Survived'].value_counts()

# print(surviced_counts[1] / surviced_counts.sum() * 100)

# 3. Какую долю пассажиры первого класса составляли среди всех пассажиров?
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).

classes = data['Pclass'].value_counts()
# print(classes[1] / classes.sum() * 100)

# 4. Какого возраста были пассажиры?
# Посчитайте среднее и медиану возраста пассажиров. В качестве ответа приведите два числа через пробел.

ages = data['Age'].dropna()
# print(ages.mean(), ' ', ages.median())

# 5. Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
corr = data['SibSp'].corr(data['Parch'])
print(corr)


# 6. Какое самое популярное женское имя на корабле?
# Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name).
# Это задание — типичный пример того, с чем сталкивается специалист по анализу данных.
# Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию.
# Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен,
# а также разделения их на женские и мужские.















names = data[data['Sex'] == 'female']['Name']
print(names)
print()

names = data[data['Sex'] == 'female']['Name'].apply(
    lambda r: r.split(',')[1].split()[1]
)
print(names.value_counts().head())

