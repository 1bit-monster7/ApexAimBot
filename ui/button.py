from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        btn = QPushButton('保存',self)
        btn.setGeometry(0, 0, 200, 100)
        btn.setToolTip('点我有惊喜')
        btn.setText('change text')




if __name__ == "__main__":
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec()
