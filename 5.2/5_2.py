import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time

# Загрузка данных
data = pd.read_csv('diabetes.csv')

# Разделение на признаки и целевую переменную
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

max_depths = range(1, 21)
train_acc = []
test_acc = []

for depth in max_depths:
    rf = RandomForestClassifier(max_depth=depth, random_state=42)
    rf.fit(X_train, y_train)
    
    train_acc.append(accuracy_score(y_train, rf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, rf.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_acc, label='Train Accuracy')
plt.plot(max_depths, test_acc, label='Test Accuracy')
plt.xlabel('Max Depth of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max Depth of Trees')
plt.legend()
plt.grid()
plt.show()

max_features = range(1, X.shape[1] + 1)
train_acc = []
test_acc = []

for features in max_features:
    rf = RandomForestClassifier(max_features=features, random_state=42)
    rf.fit(X_train, y_train)
    
    train_acc.append(accuracy_score(y_train, rf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, rf.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(max_features, train_acc, label='Train Accuracy')
plt.plot(max_features, test_acc, label='Test Accuracy')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Features')
plt.legend()
plt.grid()
plt.show()

n_estimators = [1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]
train_acc = []
test_acc = []
times = []

for n in n_estimators:
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    times.append(time.time() - start_time)
    
    train_acc.append(accuracy_score(y_train, rf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, rf.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(n_estimators, train_acc, label='Train Accuracy')
plt.plot(n_estimators, test_acc, label='Test Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Trees')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(n_estimators, times, 'r-')
plt.xlabel('Number of Trees')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time vs Number of Trees')
plt.grid()
plt.show()

# Подбор параметров для XGBoost
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
max_depths = [3, 5, 7, 9]
n_estimators = [50, 100, 150, 200]

best_acc = 0
best_params = {}

for lr in learning_rates:
    for depth in max_depths:
        for n in n_estimators:
            xgb = XGBClassifier(learning_rate=lr, max_depth=depth, n_estimators=n, random_state=42)
            xgb.fit(X_train, y_train)
            acc = accuracy_score(y_test, xgb.predict(X_test))
            
            if acc > best_acc:
                best_acc = acc
                best_params = {'learning_rate': lr, 'max_depth': depth, 'n_estimators': n}

print(f"Best accuracy: {best_acc}")
print(f"Best parameters: {best_params}")

# Обучение с лучшими параметрами
start_time = time.time()
xgb = XGBClassifier(**best_params, random_state=42)
xgb.fit(X_train, y_train)
xgb_time = time.time() - start_time
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

# Сравнение с Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=42)
start_time = time.time()
rf.fit(X_train, y_train)
rf_time = time.time() - start_time
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print("\nComparison:")
print(f"XGBoost - Accuracy: {xgb_acc:.4f}, Time: {xgb_time:.4f} sec")
print(f"Random Forest - Accuracy: {rf_acc:.4f}, Time: {rf_time:.4f} sec")

