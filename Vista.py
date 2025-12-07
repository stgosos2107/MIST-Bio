# MIST-Bio
# Mariana Ardila Alvarez
# Sofia Henao Osorio
# Valeria Salazar Ibarguen
# Santiago Osorio Salazar
#Este archivo contiene las clases de la vista (UI) de la aplicación usando PyQt5.
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


#Clases de la vista 

#La primera clase es LoginWindow y maneja la ventana de login y registro de usuarios
class LoginWindow(QDialog):
#Carga el archivo .ui, configura la interfaz y conecta los botones de login y registro con 
# sus funciones internas.
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("Interfaz/login_window.ui", self)

        self.controller = None

        if hasattr(self, "error_label"):
            self.error_label.setText("")
            self.error_label.setVisible(False)

        if hasattr(self, "login_button"):
            self.login_button.clicked.connect(self._on_login_clicked)
        elif hasattr(self, "btn_login"):
            self.btn_login.clicked.connect(self._on_login_clicked)
        elif hasattr(self, "pushButton_login"):
            self.pushButton_login.clicked.connect(self._on_login_clicked)

        if hasattr(self, "register_button"):
            self.register_button.clicked.connect(self._on_register_clicked)
        elif hasattr(self, "btn_register"):
            self.btn_register.clicked.connect(self._on_register_clicked)
        elif hasattr(self, "pushButton_register"):
            self.pushButton_register.clicked.connect(self._on_register_clicked)

#metodo que obtiene el usuario y la contraseña escritos en los campos del login.
    def get_login_data(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        return username, password

#Método alterno para obtener usuario y contraseña (llama a get_login_data())
    def get_credentials(self):
        return self.get_login_data()

#Obtiene usuario, contraseña y confirmación de contraseña;
#Verifica que los campos estén completos y que las contraseñas coincidan.
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

#Este método muestra un mensaje de error en la etiqueta correspondiente
    def show_error(self, message):
        if not hasattr(self, "error_label"):
            return
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))

#Este método limpia los campos de entrada y el mensaje de error
    def clear_fields(self):
        self.username_input.clear()
        self.password_input.clear()
        if hasattr(self, "confirm_password_input"):
            self.confirm_password_input.clear()
        self.show_error("")

#Se ejecuta cuando el usuario hace clic en “Iniciar sesión”;
#Llama al controlador para verificar credenciales y actúa según la respuesta.
    def _on_login_clicked(self):
        username, password = self.get_login_data()

        ok = True
        if self.controller is not None and hasattr(self.controller, "handle_login"):
            ok = self.controller.handle_login(username, password)

        if ok:
            self.show_error("")
            self.accept()
        else:
            self.show_error("Usuario o contraseña incorrectos.")


#Este método se ejecuta cuando el usuario hace clic en “Iniciar sesión”;
#Llama al controlador para verificar credenciales y actúa según la respuesta.
    def _on_register_clicked(self):
        data = self.get_register_data()
        if data is None:
            return

        username, password = data

        ok = True
        if self.controller is not None and hasattr(self.controller, "handle_register"):
            ok = self.controller.handle_register(username, password)

        if not ok:
            self.show_error("No se pudo registrar. Es posible que el usuario ya exista.")
            return

        frame = None
        try:
            camera_dialog = CameraCaptureDialog(self)
            frame = camera_dialog.capture_image()
        except Exception:
            frame = None

        if frame is not None:
            try:
                import os
                os.makedirs("temp", exist_ok=True)
                ruta = os.path.join("temp", f"perfil_{username}.jpg")
                cv2.imwrite(ruta, frame)
            except Exception:
                pass

        self.clear_fields()
        self.show_error("Usuario registrado. Ahora puedes iniciar sesión con tus datos.")


class MainWindow(QMainWindow):
#Carga la interfaz principal del programa y conecta el botón de cerrar sesión.
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("Interfaz/main_window.ui", self)
        self.controller = None

        if hasattr(self, "logout_button"):
            self.logout_button.clicked.connect(self._on_logout_clicked)

#Configura las tres pestañas principales de la aplicación:
#Imágenes, Señales y Datos Tabulares.
    def set_tabs(self, image_widget, signal_widget, tabular_widget):
        self.tab_widget.clear()
        self.tab_widget.addTab(image_widget, "Imágenes")
        self.tab_widget.addTab(signal_widget, "Señales")
        self.tab_widget.addTab(tabular_widget, "Datos tabulares")

#Muestra un mensaje temporal en la barra de estado.
    def update_status(self, msg, timeout_ms=5000):
        self.status_bar.showMessage(msg, timeout_ms)

#Cambia la pestaña seleccionada según el índice recibido
    def cambia_tab(self, tab_index):
        self.tab_widget.setCurrentIndex(tab_index)

#Este método ejecuta el logout llamando al controlador y luego cierra la ventana.
    def _on_logout_clicked(self):
        if self.controller is not None and hasattr(self.controller, "handle_logout"):
            self.controller.handle_logout()
        self.close()


#Esta clase maneja la vista para el procesamiento de imágenes médicas
class ImageWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("Interfaz/image_widget.ui", self)

        self.controller = None

        if hasattr(self, "load_button"):
            self.load_button.clicked.connect(self._on_load_clicked)

        if hasattr(self, "filter_button"):
            self.filter_button.clicked.connect(self._on_filter_clicked)

        if hasattr(self, "axial_slider"):
            self.axial_slider.valueChanged.connect(
                lambda value: self._on_slider_moved("axial", value)
            )
        if hasattr(self, "coronal_slider"):
            self.coronal_slider.valueChanged.connect(
                lambda value: self._on_slider_moved("coronal", value)
            )
        if hasattr(self, "sagittal_slider"):
            self.sagittal_slider.valueChanged.connect(
                lambda value: self._on_slider_moved("sagittal", value)
            )

#Permite asignar el controlador correspondiente a este widget
    def set_controller(self, controller):
        self.controller = controller

#Este método se ejecuta al presionar “Cargar imagen”;
#Llama al método del controlador que carga la imagen seleccionada
    def _on_load_clicked(self):

        if self.controller is None:
            self.get_selected_file()
            return

        self.controller.handle_load_image()

#Ejecuta la acción de aplicar un filtro a la imagen.
#Pide al controlador el resultado filtrado y lo muestra
    def _on_filter_clicked(self):

        if self.controller is None:
            return

        pixmap = self.controller.handle_process()
        if pixmap is None:
            return

        self.display_slice("axial", pixmap)

#Metodo que se ejecuta cuando un slider cambia;
#Pide el corte correspondiente (axial, coronal o sagital) al controlador.
    def _on_slider_moved(self, plane, value):
        
        if self.controller is None:
            return
        pixmap = self.controller.handle_slider_change(plane, value)
        self.display_slice(plane, pixmap)

#Muestra en pantalla la imagen recibida dentro del label correcto.
    def display_slice(self, plane, pixmap):
        plane = plane.lower()

        if plane == "axial":
            label = self.axial_label
        elif plane == "coronal":
            label = self.coronal_label
        elif plane == "sagittal":
            label = self.sagittal_label
        else:
            return

        if pixmap is None or pixmap.isNull():
            label.clear()
            return

        scaled = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setPixmap(scaled)

#Configura el rango del slider según la cantidad de cortes disponibles en el volumen
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

#Abre un diálogo para seleccionar una carpeta con una serie DICOM
    def get_selected_file(self):
        from PyQt5.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta con serie DICOM",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        return folder or ""




#Esta clase maneja la vista para el procesamiento de señales
class SignalWidget(QWidget):
#Carga la interfaz signal_widget.ui, inicializa el controlador y conecta 
# los botones (cargar señal, graficar espectro y calcular desviación estándar) con sus métodos correspondientes
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("Interfaz/signal_widget.ui", self)
        self.controller = None

        if hasattr(self, "load_button"):
            self.load_button.clicked.connect(self._on_load_clicked)
        if hasattr(self, "plot_button"):
            self.plot_button.clicked.connect(self._on_plot_clicked)
        if hasattr(self, "std_button"):
            self.std_button.clicked.connect(self._on_std_clicked)

#metodo que abre un cuadro de diálogo para que el usuario seleccione un archivo de señal.
#Permite .csv, .txt y .mat.
    def get_selected_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar señal",
            "",
            "Archivos de señal (*.csv *.txt *.mat);;Todos los archivos (*.*)",
        )
        return filename or ""

#Devuelve el nombre del canal seleccionado en el combo box.
    def get_selected_channel(self):
        return self.channel_combo.currentText()

#Devuelve el índice del canal seleccionado en el combo box
    def get_selected_channel_index(self):
        return self.channel_combo.currentIndex()

#Llena la tabla (QTableWidget) con los resultados del análisis FFT.
#Inserta los valores fila por fila y ajusta el tamaño de las columnas
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

#Convierte una figura de Matplotlib en una escena (QGraphicsScene) 
# para poder mostrarla en un QGraphicsView.
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

#Muestra un gráfico en el área correspondiente (histograma o espectro)
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

#método que llama al controlador para cargar la señal seleccionada
    def _on_load_clicked(self):
        if self.controller is None or not hasattr(self.controller, "handle_load_signal"):
            return
        self.controller.handle_load_signal()

#Solicita al controlador el espectro de la señal (FFT) y lo muestra en pantalla
    def _on_plot_clicked(self):
        if self.controller is None or not hasattr(self.controller, "handle_plot_spectrum"):
            return
        fig = self.controller.handle_plot_spectrum()
        if fig is not None:
            self.display_plot("spectrum", fig)

#metodo que Solicita al controlador el cálculo de desviación estándar y su histograma.
#Muestra la gráfica y el valor numérico en la interfaz.
    def _on_std_clicked(self):
        if self.controller is None or not hasattr(self.controller, "handle_std_dev"):
            return
        result = self.controller.handle_std_dev()
        if isinstance(result, tuple) and len(result) == 2:
            std_value, fig = result
        else:
            fig = result
            std_value = None
        if fig is not None:
            self.display_plot("histogram", fig)
        if std_value is not None and hasattr(self, "std_label"):
            self.std_label.setText(f"Desviación estándar: {std_value:.4f}")

#Esta clase maneja la vista para el procesamiento de datos tabulares
class TabularWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("Interfaz/tabular_widget.ui", self)

        self.controller = None

        self.plot_views = []
        for name in ("plot_view1", "plot_view2", "plot_view3", "plot_view4"):
            if hasattr(self, name):
                view = getattr(self, name)
                if isinstance(view, QGraphicsView):
                    self.plot_views.append(view)

        if hasattr(self, "load_csv_button"):
            self.load_csv_button.clicked.connect(self._on_load_csv_clicked)
        elif hasattr(self, "btn_load_csv"):
            self.btn_load_csv.clicked.connect(self._on_load_csv_clicked)
        elif hasattr(self, "pushButton_load_csv"):
            self.pushButton_load_csv.clicked.connect(self._on_load_csv_clicked)

        # Botón para graficar columnas
        if hasattr(self, "plot_columns_button"):
            self.plot_columns_button.clicked.connect(self._on_plot_columns_clicked)
        elif hasattr(self, "btn_plot_columns"):
            self.btn_plot_columns.clicked.connect(self._on_plot_columns_clicked)
        elif hasattr(self, "pushButton_plot_columns"):
            self.pushButton_plot_columns.clicked.connect(self._on_plot_columns_clicked)

#Permite seleccionar un archivo de datos (CSV o Excel) desde el explorador
    def get_selected_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo de datos",
            "",
            "Archivos CSV (*.csv);;Archivos Excel (*.xlsx *.xls);;Todos los archivos (*.*)",
        )
        return filename or ""

#Carga el modelo de datos (QStandardItemModel o similar) dentro del QTableView
    def load_data_model(self, model):
        self.data_table.setModel(model)

#Llena la lista de columnas disponibles para seleccionar antes de graficar
    def set_column_names(self, columns):
        self.column_list.clear()
        self.column_list.addItems(columns)

#Retorna una lista con los nombres de las columnas seleccionadas por el usuario
    def get_selected_columns(self):
        return [item.text() for item in self.column_list.selectedItems()]

#Muestra un gráfico en uno de los QGraphicsView disponibles según su posición
    def display_column_plot(self, index, figure):
        if index < 0 or index >= len(self.plot_views):
            return

        view = self.plot_views[index]
        scene = self._figure_to_scene(figure)
        view.setScene(scene)
        rect = scene.itemsBoundingRect()
        view.fitInView(rect, Qt.KeepAspectRatio)

#Convierte una figura de Matplotlib en una escena (QGraphicsScene) 
# para ser mostrada en las vistas de gráficos.
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

#método que pide al controlador que cargue el archivo CSV o Excel seleccionado
    def _on_load_csv_clicked(self):
        if self.controller is None:
            return
        self.controller.handle_load_csv()

#Pide al controlador los gráficos de las columnas seleccionadas
#  y los muestra en los distintos espacios de graficación.
    def _on_plot_columns_clicked(self):
        if self.controller is None:
            return
        figs = self.controller.handle_plot_columns()
        if not figs:
            return

        for idx, (col, fig) in enumerate(figs):
            self.display_column_plot(idx, fig)


#Esta clase maneja el diálogo para capturar imágenes desde la cámara web
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

#Toma un frame de la cámara y lo convierte a un QPixmap 
# para mostrarlo en el label de previsualización.
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

#Abre la cámara, inicia el temporizador, muestra el diálogo y espera a que el usuario presione Aceptar.
#Devuelve la imagen capturada (frame) si se aceptó.
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

#Usamos este metodo para detener el temporizador y liberar la cámara
# para evitar que quede bloqueada en segundo plano
    def _release_camera(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.video_capture is not None and self.video_capture.isOpened():
            self.video_capture.release()
        self.video_capture = None

#Se ejecuta cuando la ventana se cierra; asegura que la cámara sea liberada correctamente
    def closeEvent(self, event):
        self._release_camera()
        super().closeEvent(event)
