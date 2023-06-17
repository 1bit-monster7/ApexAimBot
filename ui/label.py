from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PySide6.QtCore import Qt


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        mainLayout = QVBoxLayout()  # 流格式

        label = QLabel('我是个标签')
        label.setText('我文字让修改了')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        mainLayout.addWidget(label)
        self.setLayout(mainLayout)


if __name__ == "__main__":
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec()
