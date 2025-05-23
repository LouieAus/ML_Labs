from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



''' ====== Подготовка набора данных ====== '''

# Загрузка набора данных по диабету
diabetes = datasets.load_diabetes()

# Преобразование в DataFrame для удобства
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Отрисовка данных на графике
plt.figure(figsize=(10, 6))
plt.scatter(df['bmi'], df['target'])
plt.xlabel('Индекс массы тела (BMI)')
plt.ylabel('Прогрессирование диабета')
plt.title('Зависимость прогрессирования диабета от индекса массы тела')
plt.show()



''' ====== Модель Scikit-Learn ====== '''

# Подготовка данных
X = df[['bmi']].values
y = df['target'].values

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)

print("\nРезультаты Scikit-Learn:")
print(f"Коэффициент (угловой коэффициент): {model_sklearn.coef_[0]}")
print(f"Интерсепт: {model_sklearn.intercept_}")
print(f"R^2 score: {model_sklearn.score(X_test, y_test)}")

# Предсказания
y_pred_sklearn = model_sklearn.predict(X_test)



''' ====== Моя модель ====== '''

class MyLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept)).dot(X_with_intercept.T).dot(y)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        
    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_

# Создание и обучение собственной модели
my_model = MyLinearRegression()
my_model.fit(X_train, y_train)

print("\nРезультаты собственной реализации:")
print(f"Коэффициент (угловой коэффициент): {my_model.coef_[0]}")
print(f"Интерсепт: {my_model.intercept_}")

# Предсказания
y_pred_my = my_model.predict(X_test)



''' ====== Отрисовка графиков ====== '''

# Создаем график
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='black', label='Реальные данные')
plt.plot(X_test, y_pred_sklearn, color='blue', linewidth=3, label='Scikit-Learn')
plt.plot(X_test, y_pred_my, color='red', linestyle='dashed', linewidth=2, label='Моя реализация')
plt.xlabel('Индекс массы тела (BMI)')
plt.ylabel('Прогрессирование диабета')
plt.title('Сравнение реализаций')
plt.legend()
plt.show()

# Создаем DataFrame с результатами
results = pd.DataFrame({
    'BMI': X_test.flatten(),
    'Actual': y_test,
    'Scikit-Learn Prediction': y_pred_sklearn,
    'My Model Prediction': y_pred_my,
    'Scikit-Learn Error': np.abs(y_test - y_pred_sklearn),
    'My Model Error': np.abs(y_test - y_pred_my)
})

# Выводим первые 20 строк
print("\nТаблица с результатами предсказаний:")
print(results.head(20))

# Выводим средние ошибки
print("\nСредняя абсолютная ошибка:")
print(f"Scikit-Learn: {results['Scikit-Learn Error'].mean()}")
print(f"Моя модель: {results['My Model Error'].mean()}")
