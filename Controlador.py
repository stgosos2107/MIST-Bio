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









