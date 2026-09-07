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

    app.setStyleSheet(app.styleSheet() + """
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            border: 1px solid #888;
            border-radius: 2px;
            background: #2b2b2b;
        }
        QCheckBox::indicator:checked {
            background: #2a84ff;
            border-color: #2a84ff;
        }
        QCheckBox::indicator:disabled {
            background: #444;
            border-color: #555;
        }
    """)

    win = MainWindow()
    win.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
