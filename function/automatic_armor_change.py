import os
import platform
import sys
import time
from pathlib import Path

import cv2
import keyboard
import numpy as np
import winsound
from PIL import ImageGrab
from PyQt5.QtWidgets import QApplication, QMainWindow
from pynput.mouse import Controller as c_mouse, Button

# 取当前py文件运行目录
from function.delay_ms import delay_ms

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def compare_images(image1, image2):
    return cv2.norm(image1, image2, cv2.NORM_L2)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.shell_locate = [(201, 773), (160, 695)]  # 护甲在死亡之箱中可能的位置
        self.sim_min = float(0.6)  # 相似度的阈值，建议0.6
        self.press_delay = float(0.001)  # 鼠标点击间隔时间(s)
        self.mouse_location = [(232, 774)]  # 换完甲后鼠标停留的位置
        self.capture_delay = float(0.2)  # 按下E键后延迟多少秒可以打开背包（一般0.5s，延迟高的话数值也调高）
        self.local_image = cv2.cvtColor(cv2.imread(ROOT / 'image' / 'local_image.png'), cv2.COLOR_BGR2GRAY)
        # 截图xy坐标长宽
        self.x = float(120)
        self.y = float(115)
        self.width = float(100)
        self.height = float(35)
        # 是否截图 调试的时候打开
        self.save_pic = False
        _, self.local_image = cv2.threshold(self.local_image, 127, 255, cv2.THRESH_BINARY_INV)
        self.mouse = c_mouse()
        keyboard.on_press_key("`", self.on_press_b)
        keyboard.on_press_key("e", self.on_press_e)
        keyboard.on_press_key("+", self.on_press_c)

    def capture_screen(self, x, y, width, height):
        screen = ImageGrab.grab(bbox=[x, y, x + width, y + height])
        if self.save_pic:
            print('从该位置截图：' + str(self.x) + ',' + str(self.y))
            print('截图长度：' + str(self.width) + ',高度' + str(self.height))
            screen.save("game_plug_in/function/image/captured_image.png")
        screen = np.array(screen)
        return cv2.cvtColor(screen, cv2.COLOR_BGR2RGB), screen

    def on_press_e(self, e):
        try:
            start_time = time.time()
            local_image = self.local_image
            while True:
                delay_ms(100)
                # Check if 'e' key is still pressed
                if not keyboard.is_pressed("e"):
                    break
                # Check if enough time has passed since last capture
                if time.time() - start_time >= self.capture_delay:
                    # Capture the image on the screen (x, y, width, height)
                    _, captured_image = self.capture_screen(self.x, self.y, self.width, self.height)
                    # Screenshot of binary processing
                    captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
                    _, captured_image = cv2.threshold(captured_image, 127, 255, cv2.THRESH_BINARY_INV)
                    # Calculate similarity
                    similarity = compare_images(captured_image, local_image)
                    similarity_percentage = (100 - similarity / 100) / 100
                    print("相似度:" + str(similarity_percentage))
                    if similarity_percentage >= self.sim_min:
                        print('背包已开启，开始换甲')
                        for x in self.shell_locate:
                            self.mouse.position = x
                            time.sleep(self.press_delay)
                            self.mouse.click(Button.left)
                            time.sleep(self.press_delay)
                            self.mouse.click(Button.left)
                        # Restore mouse position
                        self.mouse.position = self.mouse_location[0]
                        winsound.PlaySound('function/music/8855.wav', flags=1)
                    else:
                        print('背包未开启，请重试')
                        # winsound.PlaySound('function/music/ding.wav', flags=1)
                        start_time = time.time()
                    continue
        except Exception as e:
            # Handle the error and raise a new exception to indicate that an error occurred
            print("An error occurred: ", e)
            raise Exception("An error occurred during the on_press_e method.")

    def on_press_b(self, e):
        print('当前坐标是：' + str(self.mouse.position[0]) + ',' + str(
            self.mouse.position[1]))

    def on_press_c(self, e):
        self.capture_screen(self.x, self.y, self.width, self.height)


def automatic_armor_change_func():
    print('自动一键换甲')
    print('1. 按~获取当前鼠标坐标')
    print('2. 按+在指定区域截图')
    print('3. 长按E自动换甲')
    app = QApplication(sys.argv)
    win = MainWindow()
    sys.exit(app.exec_())
