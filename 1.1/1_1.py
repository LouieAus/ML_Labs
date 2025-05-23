import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class LinearRegression:
    def __init__(self):
        self.data = None
        self.x_col = 0
        self.y_col = 1
        self.a = 0
        self.b = 0

    # Функция читает данные из csv файла
    def read_data(self, filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            self.data = np.array(list(reader), dtype=float)

    # Функция выбирает нумерацию столбцов
    def set_columns(self, x_col, y_col):
        self.x_col = x_col
        self.y_col = y_col

    # Функция вычисляет статистики о данных
    def get_stats(self):            
        x_data = self.data[:, self.x_col]
        y_data = self.data[:, self.y_col]
        stats = {'x': {
                    'count': len(x_data),
                    'min': np.min(x_data),
                    'max': np.max(x_data),
                    'mean': np.mean(x_data)},
                'y': {
                    'count': len(y_data),
                    'min': np.min(y_data),
                    'max': np.max(y_data),
                    'mean': np.mean(y_data)}}
        return stats
    
    # Функция вычисляет параметры регрессионной прямой
    def calc_parameters(self):
        x = self.data[:, self.x_col]
        y = self.data[:, self.y_col]
        
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        
        self.a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        self.b = (sum_y - self.a * sum_x) / n

    def draw_data(self):
        x = self.data[:, self.x_col]
        y = self.data[:, self.y_col]
        plt.subplot(1, 3, 1)
        plt.scatter(x, y, color='blue')
        plt.title('Данные')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)

    # Функция рисует график с данными и регрессионной прямой
    def draw_line(self):
        self.calc_parameters()

        x = self.data[:, self.x_col]
        y = self.data[:, self.y_col]
        x_min, x_max = np.min(x), np.max(x)
        x_line = np.array([x_min, x_max])
        y_line = self.a * x_line + self.b

        x = self.data[:, self.x_col]
        y = self.data[:, self.y_col]
        plt.subplot(1, 3, 2)
        plt.scatter(x, y, color='blue')
        
        plt.plot(x_line, y_line, color='red', linewidth=2)
        plt.title('Регрессионная прямая')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)

    # Функция рисует заштрихованные области квадратов ошибок
    def draw_errors(self):                    
        x = self.data[:, self.x_col]
        y = self.data[:, self.y_col]
        y_pred = self.a * x + self.b
        
        plt.subplot(1, 3, 3)
        plt.scatter(x, y, color='blue')
        
        x_min, x_max = np.min(x), np.max(x)
        x_line = np.array([x_min, x_max])
        y_line = self.a * x_line + self.b
        plt.plot(x_line, y_line, color='red', linewidth=2)
        
        ax = plt.gca()
        for xi, yi, ypi in zip(x, y, y_pred):
            height = yi - ypi
            rect = Rectangle((xi, min(yi, ypi)), 0.05 * (x_max - x_min), abs(height), 
                            linewidth=1, edgecolor='green', facecolor='green', alpha=0.2)
            ax.add_patch(rect)
        
        plt.title('Квадраты ошибок')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        
    def run(self, filename, x_col=0, y_col=1):
        # Чтение данных
        self.read_data(filename)
        self.set_columns(x_col, y_col)
        
        # Вывод статистики о данных
        stats = self.get_stats()
        print("Статистика по X:")
        print("Количество: ", stats['x']['count'])
        print("Минимум: ", stats['x']['min'])
        print("Максимум: ", stats['x']['max'])
        print("Среднее: ", stats['x']['mean'], '\n')
        print("Статистика по Y:")
        print("Количество: ", stats['y']['count'])
        print("Минимум: ", stats['y']['min'])
        print("Максимум: ", stats['y']['max'])
        print("Среднее: ", stats['y']['mean'], '\n')
        
        # Построение графиков
        self.draw_data()
        self.draw_line()
        self.draw_errors()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Ввод названия файла
    filename = "student_scores.csv"
    
    # Ввод нумерации столбцов
    x_col = int(input("Введите номер столбца для X: "))
    y_col = int(input("Введите номер столбца для Y: "))
    
    # Запуск модели
    lr = LinearRegression()
    lr.run(filename, x_col, y_col)
