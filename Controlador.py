#Este archivo .py contiene la lógica del controlador para manejar las interacciones entre el modelo y 
# la vista en nuestro aplicativo MIST-Bio.

# En total tenemos 6 clases

# La primera clase es LoginController
class LoginController:
    def __init__(self, view, model):
        self.view = view
        self.model = model

    # La vista llamará este método pasando usuario y contraseña.
    def handle_login(self, username, password):
        if self.model.verify_credentials(username, password):
            return True
        return False

    # Este método recibe usuario y contraseña de la vista y le pide al modelo que registre.
    def handle_register(self, username, password):
        return self.model.register_user(username, password)

    
    

#La segunda clase es MainController
class MainController:
    def __init__(self, view, models: dict):# view:MainWindow, models: dict
        self.view = view
        self.models = models

# La vista llamará este metodo para cerrar sesión y retornar datos para registrar.
    def handle_logout(self):
        session = self.models["session"]
        info = session.end_session()
        return info

# La vista llamará este metodo para registrar actividad del usuario.
    def log_activity(self, action, path):
        user = self.models["session"].get_user()
        logger = self.models["logger"]
        logger.log_activity(user, action, path)


class ImageController:
    def __init__(self, view, model):  
        self.view = view
        self.model = model

        if hasattr(self.view, "set_controller"):
            self.view.set_controller(self)

    def handle_load_image(self):
        file_path = self.view.get_selected_file()
        if not file_path:
            return False

        ok = self.model.load_image(file_path)
        if not ok:
            return False

        for plane, slider_attr in (
            ("axial", "axial_slider"),
            ("coronal", "coronal_slider"),
            ("sagittal", "sagittal_slider"),
        ):
            max_val = self.model.get_max_slices(plane)
            slider = getattr(self.view, slider_attr, None)
            if slider is not None and max_val > 0:
                slider.setMinimum(0)
                slider.setMaximum(max_val)
                slider.setValue(max_val // 2)

                pix = self.model.get_slice(plane, slider.value())
                self.view.display_slice(plane, pix)

        return True

    def handle_slider_change(self, plane, value):
        return self.model.get_slice(plane, value)

#este método aplica un filtro a las 3 vistas y actualiza las etiquetas
    def handle_process(self):
        axial_index = self.view.axial_slider.value()
        coronal_index = self.view.coronal_slider.value()
        sagittal_index = self.view.sagittal_slider.value()

        # Usamos un filtro claro: equalización de histograma
        pix_axial, pix_coronal, pix_sagittal = self.model.apply_filter_to_slices(
            "equalize", axial_index, coronal_index, sagittal_index
        )

        # Actualizar las 3 etiquetas de la vista
        self.view.display_slice("axial", pix_axial)
        self.view.display_slice("coronal", pix_coronal)
        self.view.display_slice("sagittal", pix_sagittal)

        return pix_axial



    def get_max_slices(self, plane):
        return self.model.get_max_slices(plane)

#La cuarta clase es SignalController
class SignalController:
    def __init__(self, view, model):  # view: SignalWidget, model: SignalProcessor
        self.view = view
        self.model = model
    
    # La vista llamará este método para cargar una señal.
    def handle_load_signal(self):
        file_path = self.view.get_selected_file()
        if not file_path:
            return False

        # Cargamos el archivo .mat con las señales
        self.model.load_mat_file(file_path)
        self.model.compute_fft_all_channels()

        # Llenar la tabla con los resultados de la FFT
        df = self.model.get_fft_dataframe()
        self.view.populate_table(df)

        # Llenar el combo de canales disponibles
        self.view.channel_combo.clear()
        channels = sorted(self.model.signal_data.keys())
        for ch in channels:
            self.view.channel_combo.addItem(str(ch))

        return True
    
    # La vista pide un gráfico espectral.
    def handle_plot_spectrum(self):
        channel_index = self.view.get_selected_channel_index()
        fig = self.model.get_spectrum_plot(channel_index)
        return fig

    # La vista pide la desviación estándar y el histograma.
    def handle_std_dev(self):
        std_value, fig = self.model.calculate_std_and_histogram(axis="global")
        return std_value, fig


#La quinta clase es TabularController
class TabularController:
    def __init__(self, view, model):  # view: TabularWidget, model: TabularProcessor
        self.view = view
        self.model = model

    # La vista llamará este método para cargar un archivo CSV.
    def handle_load_csv(self):
        file_path = self.view.get_selected_file()
        if not file_path:
            return False

        self.model.load_csv(file_path)

        qt_model = self.model.get_data_model()
        columns = self.model.get_columns()

        self.view.load_data_model(qt_model)
        self.view.set_column_names(columns)

        return True

    # La vista pasará la lista de columnas seleccionadas
    def handle_plot_columns(self):
        columns = self.view.get_selected_columns()
        figs = []

        for col in columns:
            fig = self.model.get_column_plot(col)
            figs.append((col, fig))

        return figs


#La sexta clase es CameraController
class CameraController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
    

#La vista captura la imagen y la envía aquí para procesarla.
    def handle_capture(self):
        image = self.view.capture_image()
        if image is None:
            return None
        
        processed = self.model.apply_filter("grayscale", image)
        return processed
        
    

    








