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



''' ====== Машинное обучение ====== '''

# Разделяем данные на признаки (X) и целевую переменную (y)
X = data_cleaned.drop('Survived', axis=1)  # Все признаки, кроме Survived
y = data_cleaned['Survived']              # Целевая переменная (Survived)

# Разделяем на обучающую (70%) и тестовую (30%) выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создаём и обучаем модель
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test)




# Используем автоматическое определение меток
report = classification_report(y_test, y_pred)
print(report)

# Или явно указываем метки
report = classification_report(y_test, y_pred, labels=[0, 1], target_names=['Died (0)', 'Survived (1)'])
print(report)

# Расчёт матрицы ошибок
cm = confusion_matrix(y_test, y_pred)

# Визуализация
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Died (0)', 'Survived (1)'], 
            yticklabels=['Died (0)', 'Survived (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Получаем вероятности для класса 1 (Survived)
y_probs = model.predict_proba(X_test)[:, 1]

# Расчёт кривой
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Визуализация
plt.figure(figsize=(8, 4))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-кривая')
plt.grid()
plt.show()

# Расчёт FPR и TPR
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

# Визуализация
plt.figure(figsize=(8, 4))
plt.plot(fpr, tpr, label=f'ROC-кривая (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Случайная модель
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()
plt.grid()
plt.show()
