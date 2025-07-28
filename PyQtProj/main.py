import sys
import cv2
import numpy as np
import os
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QSlider, QLabel, QFileDialog, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QInputDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QRectF

class ImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Dataset Editor")
        self.images = []
        self.image_paths = []
        self.current_image_index = -1
        self.image = None
        self.original_image = None
        self.angle = 0
        self.scale = 1.0
        self.brightness = 0
        self.contrast = 1.0
        self.blur = 0
        self.rectangles = []
        self.is_drawing = False
        self.start_point = None
        self.temp_rect = None
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        # Сцена
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.image_item = None
        main_layout.addWidget(self.view)
        # Панель управления
        control_panel = QVBoxLayout()
        # Кнопка загрузки папки
        self.load_btn = QPushButton("Load Folder")
        self.load_btn.clicked.connect(self.load_folder)
        control_panel.addWidget(self.load_btn)
        # Кнопка преобразования в ч/б
        self.bw_btn = QPushButton("Convert to B/W")
        self.bw_btn.clicked.connect(self.convert_to_bw)
        control_panel.addWidget(self.bw_btn)
        # Угол поворота
        self.angle_label = QLabel("Rotation Angle: 0°")
        control_panel.addWidget(self.angle_label)
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(-180, 180)
        self.angle_slider.valueChanged.connect(self.update_image)
        control_panel.addWidget(self.angle_slider)
        # Масштабирования
        self.scale_label = QLabel("Scale: 1.0x")
        control_panel.addWidget(self.scale_label)
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(10, 200)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(self.update_image)
        control_panel.addWidget(self.scale_slider)

        # Яркости и контрастность
        self.brightness_label = QLabel("Brightness: 0")
        control_panel.addWidget(self.brightness_label)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.valueChanged.connect(self.update_image)
        control_panel.addWidget(self.brightness_slider)

        self.contrast_label = QLabel("Contrast: 1.0")
        control_panel.addWidget(self.contrast_label)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_image)
        control_panel.addWidget(self.contrast_slider)
        # Размытие
        self.blur_label = QLabel("Blur: 0")
        control_panel.addWidget(self.blur_label)
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 20)
        self.blur_slider.valueChanged.connect(self.update_image)
        control_panel.addWidget(self.blur_slider)
        # Кроп
        self.crop_btn = QPushButton("Crop Image")
        self.crop_btn.clicked.connect(self.crop_image)
        control_panel.addWidget(self.crop_btn)
        # Выделение объектов
        self.rect_btn = QPushButton("Add Rectangle")
        self.rect_btn.clicked.connect(self.toggle_rectangle_mode)
        control_panel.addWidget(self.rect_btn)
        # Отмена изменений
        self.reset_btn = QPushButton("Reset Image")
        self.reset_btn.clicked.connect(self.reset_image)
        control_panel.addWidget(self.reset_btn)

        # Создания мозаики
        self.mosaic_btn = QPushButton("Create Mosaic")
        self.mosaic_btn.clicked.connect(self.create_mosaic)
        control_panel.addWidget(self.mosaic_btn)

        # Сохранение итогового изображения
        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)
        control_panel.addWidget(self.save_btn)

        main_layout.addLayout(control_panel)

        self.view.setMouseTracking(True)
        self.scene.installEventFilter(self)
        self.setFocusPolicy(Qt.StrongFocus)
        self.view.setFocusPolicy(Qt.NoFocus)
        self.setFocus()

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.image_paths = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            self.images = []
            self.rectangles = []
            self.current_image_index = -1
            if self.image_paths:
                self.current_image_index = 0
                self.load_current_image()

    def load_current_image(self):
        if 0 <= self.current_image_index < len(self.image_paths):
            self.image = cv2.imread(self.image_paths[self.current_image_index])
            self.original_image = self.image.copy()
            self.rectangles = []
            self.scene.clear()
            self.image_item = None
            self.reset_params()
            self.update_image()

    def reset_params(self):
        self.angle = 0
        self.scale = 1.0
        self.brightness = 0
        self.contrast = 1.0
        self.blur = 0
        self.angle_slider.setValue(0)
        self.scale_slider.setValue(100)
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.blur_slider.setValue(0)
        self.blur_label.setText("Blur: 0")

    def update_image(self):
        if self.image is None:
            return

        image = self.original_image.copy()

        self.brightness = self.brightness_slider.value()
        self.contrast = self.contrast_slider.value() / 100.0
        self.brightness_label.setText(f"Brightness: {self.brightness}")
        self.contrast_label.setText(f"Contrast: {self.contrast:.1f}")
        image = cv2.convertScaleAbs(image, alpha=self.contrast, beta=self.brightness)

        self.blur = self.blur_slider.value()
        self.blur_label.setText(f"Blur: {self.blur}")
        if self.blur > 0:
            kernel_size = max(3, self.blur * 2 + 1)
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        self.scale = self.scale_slider.value() / 100.0
        self.scale_label.setText(f"Scale: {self.scale:.1f}x")
        height, width = image.shape[:2]
        new_size = (int(width * self.scale), int(height * self.scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        self.angle = self.angle_slider.value()
        self.angle_label.setText(f"Rotation Angle: {self.angle}°")
        if self.angle != 0:
            height, width = image.shape[:2]
            center = (width / 2, height / 2)
            matrix = cv2.getRotationMatrix2D(center, self.angle, 1.0)
            image = cv2.warpAffine(image, matrix, (width, height))

        self.image = image
        self.display_image(self.image)

    def display_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        if self.image_item:
            self.scene.removeItem(self.image_item)
        self.image_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
        self.scene.addItem(self.image_item)

        for rect in self.rectangles:
            self.scene.addItem(rect)

    def convert_to_bw(self):
        if self.image is not None:
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            self.update_image()

    def crop_image(self):
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        x, ok = QInputDialog.getInt(self, "Crop", "Enter X coordinate:", 0, 0, width)
        if not ok:
            return
        y, ok = QInputDialog.getInt(self, "Crop", "Enter Y coordinate:", 0, 0, height)
        if not ok:
            return
        w, ok = QInputDialog.getInt(self, "Crop", "Enter width:", width//2, 1, width-x)
        if not ok:
            return
        h, ok = QInputDialog.getInt(self, "Crop", "Enter height:", height//2, 1, height-y)
        if not ok:
            return

        self.original_image = self.original_image[y:y+h, x:x+w]
        self.rectangles = []
        self.update_image()

    def reset_image(self):
        if self.image is None:
            return
        self.original_image = cv2.imread(self.image_paths[self.current_image_index])
        self.rectangles = []
        self.scene.clear()
        self.image_item = None
        self.reset_params()
        self.update_image()

    def create_mosaic(self):
        if not self.image_paths:
            return
        num_images, ok = QInputDialog.getInt(self, "Mosaic", "Enter number of images:", 4, 1, len(self.image_paths))
        if not ok:
            return

        images = [cv2.imread(path) for path in self.image_paths[:num_images]]
        if not images:
            return

        cols = int(math.ceil(math.sqrt(num_images)))
        rows = int(math.ceil(num_images / cols))

        max_width = max(img.shape[1] for img in images)
        max_height = max(img.shape[0] for img in images)

        mosaic = np.zeros((max_height * rows, max_width * cols, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (max_width, max_height), interpolation=cv2.INTER_AREA)
            mosaic[row*max_height:(row+1)*max_height, col*max_width:(col+1)*max_width] = img_resized

        self.original_image = mosaic
        self.rectangles = []
        self.scene.clear()
        self.image_item = None
        self.reset_params()
        self.update_image()

    def toggle_rectangle_mode(self):
        self.is_drawing = not self.is_drawing
        self.rect_btn.setText("Stop Drawing" if self.is_drawing else "Add Rectangle")
        if not self.is_drawing and self.temp_rect:
            self.scene.removeItem(self.temp_rect)
            self.temp_rect = None

    def constrain_to_image(self, point):
        if self.image is None:
            return point
        height, width = self.image.shape[:2]
        x = max(0, min(point.x(), width - 1))
        y = max(0, min(point.y(), height - 1))
        return QRectF(x, y, 0, 0).topLeft()

    def normalize_rect(self, start, end):
        x1, y1 = start.x(), start.y()
        x2, y2 = end.x(), end.y()
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        return QRectF(x, y, w, h)

    def eventFilter(self, source, event):
        if source is self.scene and self.image is not None and self.is_drawing:
            if event.type() == event.GraphicsSceneMousePress:
                self.start_point = self.constrain_to_image(event.scenePos())
                self.temp_rect = QGraphicsRectItem(QRectF(self.start_point, self.start_point))
                self.temp_rect.setPen(Qt.red)
                self.scene.addItem(self.temp_rect)
                return True
            elif event.type() == event.GraphicsSceneMouseMove and self.start_point:
                end_point = self.constrain_to_image(event.scenePos())
                self.temp_rect.setRect(self.normalize_rect(self.start_point, end_point))
                return True
            elif event.type() == event.GraphicsSceneMouseRelease and self.start_point:
                end_point = self.constrain_to_image(event.scenePos())
                rect = QGraphicsRectItem(self.normalize_rect(self.start_point, end_point))
                rect.setPen(Qt.red)
                self.rectangles.append(rect)
                self.scene.removeItem(self.temp_rect)
                self.temp_rect = None
                self.scene.addItem(rect)
                self.start_point = None
                return True
        return super().eventFilter(source, event)

    def save_image(self):
        if self.image is None or self.current_image_index < 0 or self.current_image_index >= len(self.image_paths):
            return
        # Заменяет исходное изображение в папке
        file_name = self.image_paths[self.current_image_index]
        image_with_rects = self.image.copy()
        for rect in self.rectangles:
            x = int(rect.rect().x())
            y = int(rect.rect().y())
            w = int(rect.rect().width())
            h = int(rect.rect().height())
            cv2.rectangle(image_with_rects, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.imwrite(file_name, image_with_rects)
        self.original_image = image_with_rects.copy()
        self.update_image()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_W) and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
        elif event.key() in (Qt.Key_Down, Qt.Key_S) and self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.load_current_image()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = ImageEditor()
    editor.resize(1000, 600)
    editor.show()
    sys.exit(app.exec_())
