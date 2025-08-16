import os
import warnings
from FaceRqecognition import Face_MainWindow
from sys import argv,exit
from PyQt5.QtWidgets import QApplication,QMainWindow

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings(action='ignore')

if __name__ == "__main__":
    app = QApplication(argv)
    window = QMainWindow()
    ui = Face_MainWindow(window)
    window.show()
    exit(app.exec_())