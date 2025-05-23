import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



''' ====== Загрузка данных ====== '''

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

#Визуализация с помощью Matplotlib
plt.figure(figsize=(12, 5))



''' ====== Отображение зависимостей ====== '''

# График sepal length vs sepal width
plt.subplot(1, 2, 1)
for species, group in df.groupby('species'):
    plt.scatter(group['sepal length (cm)'], group['sepal width (cm)'], 
                label=species, alpha=0.7)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Length vs Sepal Width')
plt.legend()

# График petal length vs petal width
plt.subplot(1, 2, 2)
for species, group in df.groupby('species'):
    plt.scatter(group['petal length (cm)'], group['petal width (cm)'], 
                label=species, alpha=0.7)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Petal Width')
plt.legend()

plt.tight_layout()
plt.show()



''' ====== Визуализация с помощью seaborn ====== '''

sns.pairplot(df, hue='species', palette='viridis')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()



''' ====== Подготовка датасетов ====== '''

# Первый датасет: setosa и versicolor
df_setosa_versicolor = df[df['target'].isin([0, 1])]

# Второй датасет: versicolor и virginica
df_versicolor_virginica = df[df['target'].isin([1, 2])]



''' ====== Разбиение датасетов на обучающую и тестовую выборки ====== '''

# Для датасета setosa и versicolor
X_set_ver = df_setosa_versicolor.drop(['target', 'species'], axis=1)
y_set_ver = df_setosa_versicolor['target']
X_train_sv, X_test_sv, y_train_sv, y_test_sv = train_test_split(
    X_set_ver, y_set_ver, test_size=0.2, random_state=42, stratify=y_set_ver)

# Для датасета versicolor и virginica
X_ver_vir = df_versicolor_virginica.drop(['target', 'species'], axis=1)
y_ver_vir = df_versicolor_virginica['target']
X_train_vv, X_test_vv, y_train_vv, y_test_vv = train_test_split(
    X_ver_vir, y_ver_vir, test_size=0.2, random_state=42, stratify=y_ver_vir)



''' ====== Обучение ====== '''

clf_sv = LogisticRegression(random_state=0)
clf_vv = LogisticRegression(random_state=0)
clf_sv.fit(X_train_sv, y_train_sv)
clf_vv.fit(X_train_vv, y_train_vv)



''' ====== Предсказание ====== '''

y_pred_sv = clf_sv.predict(X_test_sv)
y_pred_vv = clf_vv.predict(X_test_vv)



''' ====== Вывод точности моделей ====== '''

accuracy_sv = accuracy_score(y_test_sv, y_pred_sv)
accuracy_vv = accuracy_score(y_test_vv, y_pred_vv)

print("Точность модели для setosa и versicolor:", accuracy_sv)
print("Точность модели для versicolor и virginica:", accuracy_vv)



''' ====== Генерация датасета случайным образом ====== '''

X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, random_state=1, 
                          n_clusters_per_class=1)



''' ====== Визуализация датасета ====== '''

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, 
            edgecolors='k', s=25)
plt.title('Сгенерированный датасет для бинарной классификации')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.colorbar(label='Класс')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()



''' ====== Бинарная классификация ====== '''

# Разбиение на обучающую и тестовую выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Создание и обучение модели логистической регрессии
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = clf.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")

# Визуализация разделяющей поверхности
def plot_decision_boundary(X, y, model):
    h = 0.02  # Шаг сетки
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                edgecolors='k', s=25)
    plt.title('Разделяющая поверхность логистической регрессии')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.show()

plot_decision_boundary(X, y, clf)
