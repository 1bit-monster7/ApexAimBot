from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QLineEdit
from PySide6.QtCore import Qt


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        _layout = QVBoxLayout()

        line = QLineEdit()

        line.setPlaceholderText('请输入账号')

        _layout.addWidget(line)

        self.setLayout(_layout)


if __name__ == "__main__":
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec()
