import time
from multiprocessing import set_start_method, Queue, Process, freeze_support

import cv2
import numpy as np
import pyautogui
import win32api
import win32con
import win32gui

from function.automatic_armor_change import automatic_armor_change_func
from function.config import grab_window_title
from function.configUtils import get_config_from_key
from function.grab_screen import update_hwnd_title, grab_gpt
from function.identify_firearms import find_gun_main
from function.mouse_controller import mouse_ctrl_func
from function.object_detction import load_model, interface_img
from function.web_ui import create_ui

grab_rectangle = None

grab_width = None

grab_height = None

show_window = False

top_window_name = 'win'

top_window_width = 200  # 检测窗口大小

screen_width, screen_height = pyautogui.size()


def _init_main():
    global grab_rectangle, grab_width, grab_height, top_window_name, top_window_width, show_window, screen_width, screen_height
    show_window = get_config_from_key('_is_show_top_window') == '开启'
    grab_width, grab_height = get_config_from_key('_grab_width'), get_config_from_key('_grab_height')  # 截图大小
    grab_rectangle = (
        int(screen_width / 2 - grab_width / 2), int(screen_height / 2 - grab_height / 2), grab_width,
        grab_height)  # 截图区域 xy轴


def send_nearest_pos_to_mouse_ctrl(box_list, queue, name_list):
    if not box_list:
        return None

    grab_center_x = grab_width / 2
    grab_center_y = grab_height / 2

    min_distance_sq = float('inf')
    min_position = None

    box_width = None
    box_height = None

    half_grab_width = grab_width / 2
    half_grab_height = grab_height / 2

    # 获取鼠标左键和右键的状态
    left_button_state = win32api.GetAsyncKeyState(win32con.VK_LBUTTON)
    right_button_state = win32api.GetAsyncKeyState(win32con.VK_RBUTTON)
    # 判断左键和右键是否被按下
    is_mouse_left = bool(left_button_state & 0x8000)
    is_mouse_right = bool(right_button_state & 0x8000)

    y_offset = half_grab_height * 0.11

    if left_button_state and not is_mouse_right:
        y_offset = half_grab_height * 0
    for box in box_list:
        if box[0] not in name_list:
            continue
        box_center_x = box[1] * grab_width
        box_center_y = box[2] * grab_height
        box_width = box[3] * grab_width  # 目标框宽度
        box_height = box[4] * grab_height  # 目标框高度

        distance_sq = (box_center_x - grab_center_x) ** 2 + (box_center_y - grab_center_y) ** 2
        if distance_sq < min_distance_sq:
            min_distance_sq = distance_sq
            min_position = (box_center_x - half_grab_width, box_center_y - half_grab_height - y_offset)

    if min_position is None:
        return None
    queue.put((min_position, box_width, box_height))


def draw_box(img, box_list, name_list):
    if len(box_list) == 0:
        return img
    for _box in box_list:
        if _box[0] not in name_list:
            continue
        x_center = _box[1] * grab_width
        y_center = _box[2] * grab_height
        w = _box[3] * grab_width
        h = _box[4] * grab_height
        x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
        x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=4)
        cv2.putText(img, f'{_box[0]}_{_box[5]}%', (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255),
                    2)  # box[0] 是类别名
    return img


def draw_fps(img, fps_tag, fps_list):
    timer = time.time() - fps_tag
    if len(fps_list) > 10:
        fps_list.pop(0)
        fps_list.append(timer)
    else:
        fps_list.append(timer)
    cv2.putText(img, str(int(1 / np.mean(fps_list))), (20, 100), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255), 6)
    return img


def _s(num):
    return "{:.2f}".format(num)


def detect_img(queue):
    _init_main()  # 初始化参数
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
    # 0. 初始化
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
    fps_list = []  # 记录每帧运行的时间
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
    # 1. 加载模型
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
    model, name_list, auth = load_model()

    print('\n')
    print(
        f"屏幕分辨率:{screen_width}*{screen_height}p   截图宽高:{grab_width, grab_height}   模型类别：{name_list}   游戏窗口标题:{grab_window_title}   窗口开关：{show_window}   模型名称:{auth[0]}   模型尺寸:{auth[1]}   置信度:{auth[2]}   交并集:{auth[3]}")

    while True:
        fps_tag = time.time()
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
        # 2. 截图
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
        img = grab_gpt(window_title=grab_window_title, grab_rect=grab_rectangle)
        _search_time = _s((time.time() - fps_tag) * 1000)  # 截图耗时
        # 3. 物体检测
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
        t1 = time.time()
        box_list = interface_img(img, model)
        _pred_time = _s((time.time() - t1) * 1000)  # 推理耗时
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
        # 4. 发送最近坐标
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
        send_nearest_pos_to_mouse_ctrl(box_list, queue, name_list)
        _for_time = _s((time.time() - fps_tag) * 1000)
        _fps = _s((1 / float(_for_time)) * 1000)
        # print(f"截图耗时：{_search_time}ms || 推理耗时：{_pred_time}ms ||  一次循环耗时：{_for_time}ms ||  FPS：{_fps}")

        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
        if show_window:
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            # 5. 画框
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            img = draw_box(img, box_list, name_list)

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            # 6. FPS
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            img = draw_fps(img, fps_tag, fps_list)

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            # 7. 窗口显示
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            cv2.namedWindow(top_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(top_window_name, top_window_width,
                             int(top_window_width * grab_height / grab_width))  # 重置窗口大小
            hwnd = win32gui.FindWindow(None, top_window_name)
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 1900 - top_window_width, 150, 0, 0, win32con.SWP_NOSIZE)
            cv2.imshow(top_window_name, img)
            cv2.waitKey(1)


def main():
    set_start_method('spawn')
    q = Queue()
    c = Queue()
    p_d = Process(target=detect_img, args=(q,))  # 截图
    p_m = Process(target=mouse_ctrl_func, args=(q, c))  # 鼠标移动
    p_c = Process(target=find_gun_main, args=(c,))  # 自动识别压枪
    p_auto = Process(target=automatic_armor_change_func)  # 自动换甲
    p_d.start()
    p_m.start()
    p_c.start()
    p_auto.start()

    # p_auto_shake_the_gun = Process(target=shake_gan_main)  # 抖枪宏 通用宏 但效果不如自动识别压枪宏
    # p_auto_shake_the_gun.start()


if __name__ == '__main__':
    freeze_support()  # 解决多线程 pyinstall打包bug
    update_hwnd_title()
    main()
    create_ui()  # 创建Ui界面
