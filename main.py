import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
import qdarktheme
from trafficlab.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    primary_screen = app.primaryScreen()
    if primary_screen is None:
        print("TrafficLab GUI requires an active display. No screen is currently available.")
        return 1

    app.setWindowIcon(QIcon("./media/icon.png"))

    qdarktheme.setup_theme("dark")

    win = MainWindow()
    win.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
