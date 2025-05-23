from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



''' ====== Подготовка набора данных ====== '''

# Загрузка набора данных по диабету
diabetes = datasets.load_diabetes()

# Преобразование в DataFrame для удобства
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target



''' ====== Модель Scikit-Learn ====== '''

# Подготовка данных
X = df[['bmi']].values
y = df['target'].values

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)

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

# Предсказания
y_pred_my = my_model.predict(X_test)



''' ====== Вывод метрик ====== '''

# Функция для вычисления MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Вычисление метрик для Scikit-Learn модели
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)
mape_sklearn = mean_absolute_percentage_error(y_test, y_pred_sklearn)

# Вычисление метрик для собственной модели
mae_my = mean_absolute_error(y_test, y_pred_my)
r2_my = r2_score(y_test, y_pred_my)
mape_my = mean_absolute_percentage_error(y_test, y_pred_my)

# Создаем таблицу с результатами
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'R²', 'MAPE (%)'],
    'Scikit-Learn': [mae_sklearn, r2_sklearn, mape_sklearn],
    'My Model': [mae_my, r2_my, mape_my]
})

# Выводим таблицу с округлением до 4 знаков после запятой
print(metrics_df.round(4))
