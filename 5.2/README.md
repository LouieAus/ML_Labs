# Лабораторная работа №5.2

**Лабораторную работу сделал:** Львов Илья
**Группа/Курс:** курс 3ПМ, группа 1ИП

## Описание  

Задачи к лабораторной работе:
1.	Решить задачу классификации  больных методом случайного леса.
- Провевести исследование качества модели от глубины используемых деревьев. Отрисовать зависимость на графике
-	Провести исследование качества модели от количества подаваемых на дерево признаков. Отрисовать зависимость на графике
-	Провести исследование качества модели от числа деревьев. Отрисовать на графике, дополнить  график данными  о времени обучения.
2.	Решить задачу классификации с использованием XGBoost. Исследовать время обучения, качество полученных результатов. Сравнить с данными полученными в п.1 и сделать выводы (в работе). Гиперпараметры модели подбираются в этой лабораторной в ручную. Цель – попытаться найти лучшие значения и почувствовать рабочие диапазоны параметров.
 	
## Выводы  
XGBoost показал немного лучшее качество на данном датасете.
Обе модели выделяют одинаковые наиболее важные признаки.
Random Forest проще в настройке, XGBoost требует более тонкой подстройки параметров.
Для данного датасета разница в качестве незначительна, можно использовать любую из моделей
