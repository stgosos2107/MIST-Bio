#Este archivo .py contiene la lógica del controlador para manejar las interacciones entre el modelo y 
# la vista en nuestro aplicativo MIST-Bio.

# En total tenemos 6 clases

class LoginController:
    def __init__(self, view, model):
        self.view = view
        self.model = model

#La vista llamará este metodo pasando usuario y contraseña.
    def handle_login(self, username, password):

        if self.model.verify_credentials(username, password):
            return True
        return False
    
    def connect_signals(self): #TENGO DUDA SI ESTO VA AQUI O EN LA VISTA
        self.view.login_button.clicked.connect(self.on_login_clicked)





