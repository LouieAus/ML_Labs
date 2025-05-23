from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

''' ====== Подготовка данных ====== '''

# 1. Загрузка данных
data = pd.read_csv('Titanic.csv')

# 2. Удаление строк с пропусками (1.1)
data_cleaned = data.dropna()

# 3. Удаление нечисловых столбцов, кроме Sex и Embarked (1.2)
cols_to_drop = [col for col in data_cleaned.columns 
                if data_cleaned[col].dtype == 'object' 
                and col not in ['Sex', 'Embarked']]
data_cleaned = data_cleaned.drop(columns=cols_to_drop)

# 4. Перекодировка Sex и Embarked (1.3)
data_cleaned['Sex'] = data_cleaned['Sex'].map({'male': 0, 'female': 1})
data_cleaned['Embarked'] = data_cleaned['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

# 5. Удаление PassengerId (1.4)
if 'PassengerId' in data_cleaned.columns:
    data_cleaned = data_cleaned.drop(columns=['PassengerId'])

# 6. Расчет процента потерянных данных (1.5)
initial_rows = len(data)
rows_after_cleaning = len(data_cleaned)
lost_percentage = (1 - rows_after_cleaning / initial_rows) * 100

print("Итоговые столбцы:", data_cleaned.columns)
print(f"Процент потерянных данных: {lost_percentage:.2f}%")

# Разделение на признаки и целевую переменную
X = data_cleaned.drop('Survived', axis=1)
y = data_cleaned['Survived']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Создание и обучение модели
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Предсказания
y_pred_svm = svm_model.predict(X_test)

# Метрики
print("SVM Metrics:")
print(classification_report(y_test, y_pred_svm, target_names=['Died (0)', 'Survived (1)']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("ROC-AUC:", roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1]))

from sklearn.neighbors import KNeighborsClassifier

# Создание и обучение модели
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Предсказания
y_pred_knn = knn_model.predict(X_test)

# Метрики
print("\nKNN Metrics:")
print(classification_report(y_test, y_pred_knn, target_names=['Died (0)', 'Survived (1)']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("ROC-AUC:", roc_auc_score(y_test, knn_model.predict_proba(X_test)[:, 1]))

from sklearn.linear_model import LogisticRegression

# Создание и обучение модели
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Предсказания
y_pred_lr = lr_model.predict(X_test)

# Метрики
print("\nLogistic Regression Metrics:")
print(classification_report(y_test, y_pred_lr, target_names=['Died (0)', 'Survived (1)']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1]))

# Получение вероятностей для всех моделей
y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
y_proba_svm = svm_model.predict_proba(X_test)[:, 1]
y_proba_knn = knn_model.predict_proba(X_test)[:, 1]

# Построение ROC-кривых
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_proba_lr):.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_score(y_test, y_proba_svm):.2f})')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_score(y_test, y_proba_knn):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые моделей')
plt.legend()
plt.show()
