#Este archivo .py contiene la lógica del controlador para manejar las interacciones entre el modelo y 
# la vista en nuestro aplicativo MIST-Bio.

# En total tenemos 6 clases

#La primer clase es LoginController
class LoginController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
        self.connect_signals()

    def connect_signals(self):
        self.view.login_button.clicked.connect(self.handle_login)

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
        self.connect_signals()

    
    def connect_signals(self):
        self.view.logout_button.clicked.connect(self.handle_logout)

    def handle_logout(self):
        session = self.models["session"]
        session_data = session.end_session()

        logger = self.models["logger"]
        logger.log_activity(session_data["user"], "logout", "-")

        self.view.close()

    def log_activity(self, action: str, result_path: str):
        user = self.models["session"].get_user()
        logger = self.models["logger"]
        logger.log_activity(user, action, result_path)





