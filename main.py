from vista import *
from modelo import *
from controlador import *
import sys
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)

# Modelos
auth = AuthManager()
logger = DatabaseLogger()

# Vista
login_view = LoginWindow()
main_view = MainWindow()

# Controladores
login_ctrl = LoginController(login_view, auth)

login_view.show()
sys.exit(app.exec_())