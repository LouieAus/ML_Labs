import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, precision_recall_curve, 
                             roc_curve, average_precision_score)
from sklearn.tree import plot_tree
import seaborn as sns

# Загрузка данных
data = pd.read_csv('diabetes.csv')

# Разделение на признаки и целевую переменную
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Логистическая регрессия
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

# Решающее дерево (стандартные параметры)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_proba_dt = dt.predict_proba(X_test)[:, 1]

# Функция для вывода метрик
def print_metrics(y_true, y_pred, y_proba, model_name):
    print(f"\nМетрики для {model_name}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_proba))
    print("\nМатрица ошибок:")
    print(confusion_matrix(y_true, y_pred))
    print("\nОтчет классификации:")
    print(classification_report(y_true, y_pred))

# Вывод метрик
print_metrics(y_test, y_pred_lr, y_proba_lr, "Логистической регрессии")
print_metrics(y_test, y_pred_dt, y_proba_dt, "Решающего дерева")

# Исследуем F1-score как компромисс между precision и recall
max_depths = range(1, 21)
f1_scores = []

for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(max_depths, f1_scores, marker='o')
plt.title('Зависимость F1-score от глубины решающего дерева')
plt.xlabel('Глубина дерева')
plt.ylabel('F1-score')
plt.grid(True)
plt.show()

# Оптимальная глубина
optimal_depth = max_depths[np.argmax(f1_scores)]
print(f"Оптимальная глубина дерева: {optimal_depth}")

# Создание и обучение модели с оптимальной глубиной
dt_optimal = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
dt_optimal.fit(X_train, y_train)

# Визуализация дерева
plt.figure(figsize=(20, 10))
plot_tree(dt_optimal, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], 
          filled=True, rounded=True, proportion=True)
plt.title("Решающее дерево для классификации диабета")
plt.show()

# Важность признаков
importances = dt_optimal.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Важность признаков")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()

# PR и ROC кривые
# PR кривая
precision, recall, _ = precision_recall_curve(y_test, dt_optimal.predict_proba(X_test)[:, 1])
plt.figure(figsize=(10, 5))
plt.plot(recall, precision, marker='.')
plt.title('PR кривая')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()

# ROC кривая
fpr, tpr, _ = roc_curve(y_test, dt_optimal.predict_proba(X_test)[:, 1])
plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, marker='.')
plt.title('ROC кривая')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

# Исследуем влияние max_features на F1-score
max_features = range(1, X.shape[1]+1)
f1_scores_features = []

for n in max_features:
    dt = DecisionTreeClassifier(max_depth=optimal_depth, max_features=n, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores_features.append(f1)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(max_features, f1_scores_features, marker='o')
plt.title('Зависимость F1-score от max_features')
plt.xlabel('max_features')
plt.ylabel('F1-score')
plt.grid(True)
plt.show()
