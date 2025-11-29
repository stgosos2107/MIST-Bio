#Creación de las clases que harán parte del controlador
class LoginController:
    def __init__(self, view, model):
        self.view = view          # LoginWindow
        self.model = model        # AuthManager

    def handle_login(self):
        pass

    def connect_signals(self):
        pass



class MainController:
    def __init__(self, view, session, controllers: dict):
        self.view = view              # MainWindow
        self.session = session        # UserSession
        self.controllers = controllers  # dict[str, Controller]

    def handle_logout(self):
        pass

    def log_activity(self, action: str):
        pass



class ImageController:
    def __init__(self, view, model):
        self.view = view              # ImageViewerWidget
        self.model = model            # ImageProcessor

    def handle_load_image(self):
        pass

    def handle_slider_change(self, plane: str, value: int):
        pass

    def handle_apply_filter(self):
        pass



class SignalController:
    def __init__(self):
        self.attribute1 = None   # defaultValue
        self.attribute2 = None
        self.attribute3 = None

    def operation1(self, params):
        pass

    def operation2(self, params):
        pass

    def operation3(self):
        pass
