import sys
from PyQt5.QtWidgets import QApplication, QDialog

from Vista import *
from Modelo import *
from Controlador import *


def main():
    app = QApplication(sys.argv)

    # Modelos "globales"
    auth = AuthManager()
    logger = DatabaseLogger()

    # ----- LOGIN -----
    login_view = LoginWindow()
    login_ctrl = LoginController(login_view, auth)
    login_view.controller = login_ctrl

    # Ejecutar login como diálogo
    result = login_view.exec_()

    if result == QDialog.Accepted:
        # Usuario que inició sesión
        username, _ = login_view.get_credentials()
        session = UserSession(username)

        # ----- MAIN WINDOW -----
        main_view = MainWindow()

        models_for_main = {
            "session": session,
            "logger": logger,
        }
        main_ctrl = MainController(main_view, models_for_main)
        main_view.controller = main_ctrl

        # ----- MODELOS ESPECÍFICOS -----
        image_model = ImageProcessor()
        signal_model = SignalProcessor()
        tabular_model = TabularProcessor()

        # ----- VISTAS DE LOS TABS -----
        image_widget = ImageWidget()
        signal_widget = SignalWidget()
        tabular_widget = TabularWidget()

        # ----- CONTROLADORES DE LOS TABS -----
        image_ctrl = ImageController(image_widget, image_model)
        signal_ctrl = SignalController(signal_widget, signal_model)
        tabular_ctrl = TabularController(tabular_widget, tabular_model)

        # Conectar vistas con sus controladores
        image_widget.controller = image_ctrl
        signal_widget.controller = signal_ctrl
        tabular_widget.controller = tabular_ctrl

        # Poner los widgets en las pestañas del main
        main_view.set_tabs(image_widget, signal_widget, tabular_widget)

        # Mostrar ventana principal
        main_view.show()
        sys.exit(app.exec_())
    else:
        # Si cierra el login sin autenticarse
        sys.exit(0)


if __name__ == "__main__":
    main()
