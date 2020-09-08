# Импортируем библиотеки
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")


def pprint(smth):
    print(smth)
    print()


# Загружаем данные
titanic = sns.load_dataset('titanic')

# Первые 10 значений
pprint(titanic.head(10))

# Сколько столбцов и строк
pprint(titanic.shape)

pprint(titanic.describe())

# Сколько выживших
pprint(titanic['survived'].value_counts())

# График выживших
sns.countplot(titanic['survived'], label="Count")

# График выживших по столбцам 'who', 'sex', 'pclass', 'sibsp', 'parch', 'embarked'
cols = ['who', 'sex', 'pclass', 'sibsp', 'parch', 'embarked']

n_rows = 2
n_cols = 3

fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.2))

for r in range(0, n_rows):
    for c in range(0, n_cols):
        i = r * n_cols + c  # index to go through the number of columns
        ax = axs[r][c]      # Show where to position each subplot
        sns.countplot(titanic[cols[i]], hue=titanic["survived"], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title="survived", loc='upper right')

plt.tight_layout()  # tight_layout

# По полу
pprint(titanic.groupby('sex')[['survived']].mean())

# По полу и классу
pprint(titanic.pivot_table('survived', index='sex', columns='class'))

# График
titanic.pivot_table('survived', index='sex', columns='class').plot()

# График по классам
sns.barplot(x='class', y='survived', data=titanic)

# Кто больше заплатил по классам
plt.scatter(titanic['fare'], titanic['class'],  color='purple', label='Passenger Paid')
plt.ylabel('Class')
plt.xlabel('Price / Fare')
plt.title('Price Of Each Class')
plt.legend()

# Количество отсутствующих данных по колонкам
pprint(titanic.isna().sum())

# Посчитаем по всем значениям
for val in titanic:
    print(titanic[val].value_counts())
    print()

# Удалим ненужные стоблцы
titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'alone', 'adult_male', 'who'], axis=1)

# Удалим пропущенные значения
titanic = titanic.dropna(subset=['embarked', 'age'])

# Какая теперь форма?
pprint(titanic.shape)

# Какие есть типы данных?
pprint(titanic.dtypes)

# Уникальные значения по столбцам
print(titanic['sex'].unique())
print(titanic['embarked'].unique())

# Трансформируем категоральные признаки
labelencoder = LabelEncoder()

titanic.iloc[:, 2] = labelencoder.fit_transform(titanic.iloc[:, 2].values)
# print(labelencoder.fit_transform(titanic.iloc[:,2].values))

titanic.iloc[:, 7] = labelencoder.fit_transform(titanic.iloc[:, 7].values)
# print(labelencoder.fit_transform(titanic.iloc[:,7].values))

# Новые уникальные значения
print(titanic['sex'].unique())
print(titanic['embarked'].unique())

# Разделим данные на независимую часть 'X' и зависимую часть 'Y' переменных
X = titanic.iloc[:, 1:8].values
Y = titanic.iloc[:, 0].values

# Разделим данные на 80% обучающей выборки и 20% тестовой выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Преобразуем размер данных стандартных скейлером
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Создадим модели машинного обучения
def models(X_train, Y_train):
    # Using Logistic Regression Algorithm to the Training Set
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)

    # Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(X_train, Y_train)

    # Using SVC method of svm class to use Support Vector Machine Algorithm
    from sklearn.svm import SVC
    svc_lin = SVC(kernel='linear', random_state=0)
    svc_lin.fit(X_train, Y_train)

    # Using SVC method of svm class to use Kernel SVM Algorithm
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf', random_state=0)
    svc_rbf.fit(X_train, Y_train)

    # Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)

    # Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)

    # Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(X_train, Y_train)

    # print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
    print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
    print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))

    return log, knn, svc_lin, svc_rbf, gauss, tree, forest


# Натренируем все наши модели
model = models(X_train, Y_train)

for i in range(len(model)):
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    # extracting TN, FP, FN, TP
    TN, FP, FN, TP = confusion_matrix(Y_test, model[i].predict(X_test)).ravel()
    print(cm)
    print('Model[{}] Testing Accuracy = "{} !"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))
    print()

# Узнаем важность свойств
forest = model[6]
importances = pd.DataFrame({'feature': titanic.iloc[:, 1:8].columns,
                            'importance': np.round(forest.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending=False).set_index('feature')
pprint(importances)

# Отобразим на графике
importances.plot.bar()

# Предсказание модели Random Forest Classifier
pred = model[6].predict(X_test)
pprint(pred)

# Печать реальных значений
pprint(Y_test)

my_survival = [[25, 71.2833, 1, 2, 0, 0, 1]]
# Предсказание модели Random Forest Classifier
pred = model[6].predict(my_survival)
print(pred)

if pred == 0:
    print('''Не выжил ((((''')
else:
    print('Выжил')
