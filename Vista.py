from PyQt5 import uic
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QDialog,
    QFileDialog,
    QTableWidgetItem,
    QGraphicsScene,
    QGraphicsView,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer 
import Interfaz.recursos_rc
import pandas as pd
from matplotlib.figure import Figure
import cv2


class LoginWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("Interfaz/login_window.ui", self)
        if hasattr(self, "error_label"):
            self.error_label.setText("")
            self.error_label.setVisible(False)

    def get_login_data(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        return username, password

    def get_credentials(self):
        return self.get_login_data()

    def get_register_data(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        confirm = self.confirm_password_input.text()

        if not username or not password or not confirm:
            self.show_error("Por favor completa todos los campos.")
            return None

        if password != confirm:
            self.show_error("Las contraseñas no coinciden.")
            return None

        self.show_error("")
        return username, password

    def show_error(self, message):
        if not hasattr(self, "error_label"):
            return
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))

    def clear_fields(self):
        self.username_input.clear()
        self.password_input.clear()
        if hasattr(self, "confirm_password_input"):
            self.confirm_password_input.clear()
        self.show_error("")


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("Interfaz/main_window.ui", self)

    def set_tabs(self, image_widget, signal_widget, tabular_widget):
        self.tab_widget.clear()
        self.tab_widget.addTab(image_widget, "Imágenes")
        self.tab_widget.addTab(signal_widget, "Señales")
        self.tab_widget.addTab(tabular_widget, "Datos tabulares")

    def update_status(self, msg, timeout_ms=5000):
        self.status_bar.showMessage(msg, timeout_ms)

    def cambia_tab(self, tab_index):
        self.tab_widget.setCurrentIndex(tab_index)


class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("Interfaz/image_widget.ui", self)

    def display_slice(self, plane, pixmap):
        plane = plane.lower()
        if plane == "axial":
            self.axial_label.setPixmap(pixmap)
        elif plane == "coronal":
            self.coronal_label.setPixmap(pixmap)
        elif plane == "sagittal":
            self.sagittal_label.setPixmap(pixmap)

    def set_slider_range(self, plane, max_value):
        plane = plane.lower()
        if plane == "axial":
            slider = self.axial_slider
        elif plane == "coronal":
            slider = self.coronal_slider
        elif plane == "sagittal":
            slider = self.sagittal_slider
        else:
            return
        slider.setMinimum(0)
        slider.setMaximum(max_value)

    def get_selected_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen médica",
            "",
            "Imágenes médicas (*.dcm *.nii *.nii.gz);;Imágenes comunes (*.png *.jpg *.jpeg)",
        )
        return filename or ""


class SignalWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("Interfaz/signal_widget.ui", self)

    def get_selected_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar señal",
            "",
            "Archivos de señal (*.csv *.txt *.mat);;Todos los archivos (*.*)",
        )
        return filename or ""

    def get_selected_channel(self):
        return self.channel_combo.currentText()

    def get_selected_channel_index(self):
        return self.channel_combo.currentIndex()

    def populate_table(self, data):
        self.fft_table.clear()
        self.fft_table.setRowCount(len(data.index))
        self.fft_table.setColumnCount(len(data.columns))
        self.fft_table.setHorizontalHeaderLabels(list(data.columns))

        for i, row in enumerate(data.itertuples(index=False)):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.fft_table.setItem(i, j, item)

        self.fft_table.resizeColumnsToContents()

    def _figure_to_scene(self, figure):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        width, height = figure.canvas.get_width_height()
        buf = canvas.buffer_rgba()
        image = QImage(buf, width, height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(image)

        scene = QGraphicsScene(self)
        scene.addPixmap(pixmap)
        return scene

    def display_plot(self, view_type, figure):
        scene = self._figure_to_scene(figure)

        if view_type == "histogram":
            view = self.histogram_view
        elif view_type == "spectrum":
            view = self.spectrum_view
        else:
            return

        view.setScene(scene)
        rect = scene.itemsBoundingRect()
        view.fitInView(rect, Qt.KeepAspectRatio)


class TabularWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("Interfaz/tabular_widget.ui", self)

        self.plot_views = []
        for name in ("plot_view1", "plot_view2", "plot_view3", "plot_view4"):
            if hasattr(self, name):
                view = getattr(self, name)
                if isinstance(view, QGraphicsView):
                    self.plot_views.append(view)

    def get_selected_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo de datos",
            "",
            "Archivos CSV (*.csv);;Archivos Excel (*.xlsx *.xls);;Todos los archivos (*.*)",
        )
        return filename or ""

    def load_data_model(self, model):
        self.data_table.setModel(model)

    def set_column_names(self, columns):
        self.column_list.clear()
        self.column_list.addItems(columns)

    def get_selected_columns(self):
        return [item.text() for item in self.column_list.selectedItems()]

    def display_column_plot(self, index, figure):
        if index < 0 or index >= len(self.plot_views):
            return

        view = self.plot_views[index]
        scene = self._figure_to_scene(figure)
        view.setScene(scene)
        rect = scene.itemsBoundingRect()
        view.fitInView(rect, Qt.KeepAspectRatio)

    def _figure_to_scene(self, figure):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        width, height = figure.canvas.get_width_height()
        buf = canvas.buffer_rgba()
        image = QImage(buf, width, height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(image)

        scene = QGraphicsScene(self)
        scene.addPixmap(pixmap)
        return scene

class CameraCaptureDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("Interfaz/camera_capture_dialog.ui", self)

        self.video_capture = None
        self.current_frame = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)

        if hasattr(self, "capture_button"):
            self.capture_button.clicked.connect(self.accept)
        if hasattr(self, "cancel_button"):
            self.cancel_button.clicked.connect(self.reject)

    def update_preview(self):
        if self.video_capture is None or not self.video_capture.isOpened():
            return

        ok, frame = self.video_capture.read()
        if not ok:
            return

        self.current_frame = frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        if hasattr(self, "preview_label"):
            self.preview_label.setPixmap(
                pixmap.scaled(
                    self.preview_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def capture_image(self):
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.video_capture.isOpened():
            return None

        self.current_frame = None
        self.timer.start(30)

        result = None
        if self.exec_() == QDialog.Accepted and self.current_frame is not None:
            result = self.current_frame.copy()

        self._release_camera()
        return result

    def _release_camera(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.video_capture is not None and self.video_capture.isOpened():
            self.video_capture.release()
        self.video_capture = None

    def closeEvent(self, event):
        self._release_camera()
        super().closeEvent(event)
