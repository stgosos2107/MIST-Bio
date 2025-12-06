#Este archivo .py contiene la lógica del controlador para manejar las interacciones entre el modelo y 
# la vista en nuestro aplicativo MIST-Bio.

# En total tenemos 6 clases

#La primer clase es LoginController
class LoginController:
    def __init__(self, view, model):
        self.view = view
        self.model = model

#La vista llamará este metodo pasando usuario y contraseña.
    def handle_login(self, username, password):

        if self.model.verify_credentials(username, password):
            return True
        return False
    
    

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



#La tercer clase es ImageController
class ImageController:
    def __init__(self, view, model):#view: ImageViewerWidget, model: ImageProcessor
        self.view = view
        self.model = model

# La vista llamará este metodo para cargar una imagen.
    def handle_load_image(self):
        file_path = self.view.get_selected_file()
        if file_path:
            self.model.load_image(file_path)
            return True
        return False
    
#La vista pasará el plano e índice.
    def handle_slider_change(self, plane, value):
        return self.model.get_slice(plane, value)
    
# La vista llamará este metodo para aplicar un filtro.
    def handle_process(self):
        return self.model.apply_filter("default")

# La vista pasará el plano para obtener el número máximo de slices. METODO ADICIONAL
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
        
    







