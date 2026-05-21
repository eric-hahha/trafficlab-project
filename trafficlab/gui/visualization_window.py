from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow

from .tabs.tab_visualization import VisualizationTab


class VisualizationWindow(QMainWindow):
    def __init__(self, initial_file=None):
        super().__init__()
        self.setWindowTitle("TrafficLab Visualization")
        self.setWindowFlags(Qt.Window)
        self.resize(1600, 960)

        self.viewer = VisualizationTab()
        self.setCentralWidget(self.viewer)

        if initial_file:
            self.viewer.load_file(initial_file)
