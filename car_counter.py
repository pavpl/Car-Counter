import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Button, Label, Frame, LEFT, RIGHT, TOP, BOTTOM, BOTH, X, Y, Canvas

class ObjectCounter:
    def __init__(self):
        self.model = YOLO('yolov8x.pt')
        self.vehicle_classes = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
        }

    def resize_frame(self, frame, target_width=700):
        h, w = frame.shape[:2]
        if w > target_width:
            scale = target_width / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        return frame

    def count_on_image(self, image_path, scale=2):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл {image_path} не найден")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение {image_path}")
        if scale != 1:
            image_scaled = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        else:
            image_scaled = image.copy()
        results = self.model(image_scaled)
        counts = {k: 0 for k in self.vehicle_classes.values()}
        counts['total'] = 0
        image_with_boxes = image_scaled.copy()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in self.vehicle_classes:
                    obj_type = self.vehicle_classes[cls]
                    counts[obj_type] += 1
                    counts['total'] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0) if obj_type == 'person' else (0, 0, 255)
                    label = f"{obj_type} {conf:.2f}"
                    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image_with_boxes, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return image_scaled, image_with_boxes, counts

    def count_on_video_gui(self, video_path, target_width=700, frame_step=6):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Файл {video_path} не найден")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        # Считываем все нужные кадры заранее (можно оптимизировать для больших видео)
        for i in range(0, total_frames, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.resize_frame(frame, target_width)
            frames.append(frame)
        cap.release()
        # Обработка кадров и запуск GUI
        VideoPlayer(frames, self.model, self.vehicle_classes)

class VideoPlayer:
    def __init__(self, frames, model, vehicle_classes):
        self.frames = frames
        self.model = model
        self.vehicle_classes = vehicle_classes
        self.idx = 0
        self.paused = True
        self.root = Tk()
        self.root.title('Видео-анализ')
        self.canvas = Canvas(self.root, width=frames[0].shape[1], height=frames[0].shape[0])
        self.canvas.pack()
        self.info_label = Label(self.root, text='', font=("Arial", 12))
        self.info_label.pack()
        btn_frame = Frame(self.root)
        btn_frame.pack(fill=X)
        self.btn_start = Button(btn_frame, text='|< В начало', command=self.to_start)
        self.btn_start.pack(side=LEFT, padx=5, pady=5)
        self.btn_back = Button(btn_frame, text='<< Назад', command=self.prev_frame)
        self.btn_back.pack(side=LEFT, padx=5, pady=5)
        self.btn_pause = Button(btn_frame, text='Пауза/Старт', command=self.toggle_pause)
        self.btn_pause.pack(side=LEFT, padx=5, pady=5)
        self.btn_next = Button(btn_frame, text='Вперёд >>', command=self.next_frame)
        self.btn_next.pack(side=LEFT, padx=5, pady=5)
        self.btn_end = Button(btn_frame, text='В конец >|', command=self.to_end)
        self.btn_end.pack(side=LEFT, padx=5, pady=5)
        self.btn_save = Button(btn_frame, text='Сохранить кадр', command=self.save_frame)
        self.btn_save.pack(side=LEFT, padx=5, pady=5)
        self.update_frame()
        self.root.after(100, self.play_loop)
        self.root.mainloop()

    def process_frame(self, frame):
        results = self.model(frame)
        counts = {k: 0 for k in self.vehicle_classes.values()}
        counts['total'] = 0
        frame_disp = frame.copy()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in self.vehicle_classes:
                    obj_type = self.vehicle_classes[cls]
                    counts[obj_type] += 1
                    counts['total'] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0) if obj_type == 'person' else (0, 0, 255)
                    label = f"{obj_type} {conf:.2f}"
                    cv2.rectangle(frame_disp, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_disp, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame_disp, counts

    def update_frame(self):
        frame = self.frames[self.idx]
        frame_disp, counts = self.process_frame(frame)
        img_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.img_tk = img_tk
        self.canvas.create_image(0, 0, anchor='nw', image=img_tk)
        info = f"Кадр: {self.idx+1}/{len(self.frames)} | Объектов: {counts['total']} | Cars: {counts['car']} | Persons: {counts['person']} | Moto: {counts['motorcycle']} | Bus: {counts['bus']} | Truck: {counts['truck']}"
        self.info_label.config(text=info)

    def play_loop(self):
        if not self.paused:
            self.idx = (self.idx + 1) % len(self.frames)
            self.update_frame()
        self.root.after(200, self.play_loop)

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.config(text='Продолжить' if self.paused else 'Пауза')

    def next_frame(self):
        self.paused = True
        self.btn_pause.config(text='Продолжить')
        self.idx = min(len(self.frames) - 1, self.idx + 1)
        self.update_frame()

    def prev_frame(self):
        self.paused = True
        self.btn_pause.config(text='Продолжить')
        self.idx = max(0, self.idx - 1)
        self.update_frame()

    def to_start(self):
        self.paused = True
        self.btn_pause.config(text='Продолжить')
        self.idx = 0
        self.update_frame()

    def to_end(self):
        self.paused = True
        self.btn_pause.config(text='Продолжить')
        self.idx = len(self.frames) - 1
        self.update_frame()

    def save_frame(self):
        frame = self.frames[self.idx]
        frame_disp, _ = self.process_frame(frame)
        save_path = filedialog.asksaveasfilename(defaultextension='.jpg', filetypes=[('JPEG', '*.jpg'), ('PNG', '*.png')], title='Сохранить кадр')
        if save_path:
            cv2.imwrite(save_path, frame_disp[:, :, ::-1])

def main():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title='Выберите изображение или видео', filetypes=[('Image/Video Files', '*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv')])
    root.destroy()
    if not file_path:
        print('Файл не выбран. Завершение работы.')
        return
    counter = ObjectCounter()
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            scale = 2
            orig_img, boxed_img, counts = counter.count_on_image(file_path, scale=scale)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            boxed_img = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(orig_img)
            plt.title('Исходное изображение (увеличено)')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(boxed_img)
            plt.title(f'Обнаружено: {counts["total"]}\nЛегковые: {counts["car"]}, Мотоциклы: {counts["motorcycle"]}, Автобусы: {counts["bus"]}, Грузовики: {counts["truck"]}, Людей: {counts["person"]}')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            print('Обработка видео...')
            frame_step = 6
            counter.count_on_video_gui(file_path, target_width=700, frame_step=frame_step)
        else:
            print('Неподдерживаемый формат файла.')
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main() 