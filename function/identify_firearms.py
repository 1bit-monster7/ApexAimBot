import os

import cv2
import numpy as np
import win32api
import win32con
from PIL import ImageGrab

import G
from function.delay_ms import delay_ms
from function.grab_screen import grab_screen_win32_v2

pressed = False

active_weapon = None

find_count = 0

grab_window_title = 'Apex Legends'


def is_key_pressed(key_code):
    """检查指定的按键是否被按下"""
    return win32api.GetAsyncKeyState(key_code) < 0


def is_left_click():
    return is_key_pressed(win32con.VK_LBUTTON)


def is_right_click():
    return is_key_pressed(win32con.VK_RBUTTON)


def find_gun_main(c):
    global active_weapon, find_count
    while True:
        if not is_left_click() and not is_right_click():
            for gun_name in G.recoil_patterns:
                if search_image_binarization(gun_name) is not None:
                    # 如果找到了新的枪械，则更新 active_weapon 变量
                    if active_weapon == gun_name:
                        break
                    active_weapon = gun_name
                    c.put(active_weapon)
                    print(f"已更新枪械：{gun_name}")
                    break
            else:
                find_count += 1
                # print(f"未找到枪械第{find_count}次")
                if find_count == 5:
                    active_weapon = None
                    find_count = 0
                    # print('5次未找到枪械清空当前枪械', active_weapon)
        delay_ms(400)


def capture_screen(x, y, w, h):
    screen = ImageGrab.grab(bbox=[x, y, x + w, y + h])
    screen = np.array(screen)
    return cv2.cvtColor(screen, cv2.COLOR_BGR2RGB), screen


def compare_images(image1, image2):
    size = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0]))
    image1_resized = cv2.resize(image1, size)
    image2_resized = cv2.resize(image2, size)
    distance = cv2.norm(image1_resized, image2_resized, cv2.NORM_L2)
    return distance


def search_image_binarization(gun_name, confidence=0.8):
    image_path = 'function/image/gun/{}.png'.format(gun_name)
    if os.path.exists(image_path):
        # 二值化截图
        captured_image = grab_screen_win32_v2(window_title=grab_window_title, grab_rect=(G.left, G.top, G.width, G.height))
        captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)  # 灰度
        _, captured_image = cv2.threshold(captured_image, 127, 255, cv2.THRESH_BINARY_INV)  # 二值化

        local_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)  # 灰度
        _, local_image = cv2.threshold(local_image, 127, 255, cv2.THRESH_BINARY_INV)  # 二值化

        similarity = compare_images(captured_image, local_image)
        similarity_percentage = (100 - similarity / 100) / 100
        if similarity_percentage > confidence:
            return True
        else:
            return None
    else:
        # print(gun_name, '图像文件不存在！')
        return None
