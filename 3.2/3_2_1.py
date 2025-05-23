import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



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

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Удаляем Embarked и повторяем обучение
X_no_embarked = X.drop('Embarked', axis=1)
X_train_no_emb, X_test_no_emb, y_train_no_emb, y_test_no_emb = train_test_split(
    X_no_embarked, y, test_size=0.3, random_state=42
)

model_no_emb = LogisticRegression(max_iter=1000, random_state=42)
model_no_emb.fit(X_train_no_emb, y_train_no_emb)
y_pred_no_emb = model_no_emb.predict(X_test_no_emb)

accuracy_no_emb = accuracy_score(y_test_no_emb, y_pred_no_emb)
print(f"Точность без Embarked: {accuracy_no_emb:.4f} ({accuracy_no_emb * 100:.2f}%)")

# Сравнение
print(f"Изменение точности: {accuracy - accuracy_no_emb:.4f}")
