import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

class CarCounter:
    def __init__(self):
        # Используем самую точную модель YOLOv8x
        self.model = YOLO('yolov8x.pt')
        
        # Словарь для классификации транспортных средств
        self.vehicle_classes = {
            2: 'car',           # легковой автомобиль
            3: 'motorcycle',    # мотоцикл
            5: 'bus',          # автобус
            7: 'truck',        # грузовик
        }
        
    def count_vehicles(self, image_path, scale=2):
        # Проверяем существование файла
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл {image_path} не найден")
            
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение {image_path}")
            
        # Масштабируем изображение
        if scale != 1:
            image_scaled = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        else:
            image_scaled = image.copy()
        
        # Получаем предсказания от модели
        results = self.model(image_scaled)
        
        # Счетчики для разных типов транспортных средств
        vehicle_counts = {
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0,
            'total': 0
        }
        
        # Обрабатываем результаты
        image_with_boxes = image_scaled.copy()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in self.vehicle_classes:
                    vehicle_type = self.vehicle_classes[cls]
                    vehicle_counts[vehicle_type] += 1
                    vehicle_counts['total'] += 1
                    
                    # Рисуем рамку вокруг обнаруженного транспортного средства
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image_with_boxes, vehicle_type, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        return image_scaled, image_with_boxes, vehicle_counts

def main():
    # Открываем диалог выбора файла
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title='Выберите изображение', filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp')])
    root.destroy()
    if not file_path:
        print('Файл не выбран. Завершение работы.')
        return
    counter = CarCounter()
    try:
        # Масштаб увеличения (можно менять)
        scale = 2
        orig_img, boxed_img, counts = counter.count_vehicles(file_path, scale=scale)
        # Переводим BGR в RGB для корректного отображения в matplotlib
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        boxed_img = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.title('Исходное изображение (увеличено)')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(boxed_img)
        plt.title(f'Обнаружено авто: {counts["total"]}\nЛегковые: {counts["car"]}, Мотоциклы: {counts["motorcycle"]}, Автобусы: {counts["bus"]}, Грузовики: {counts["truck"]}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main() 