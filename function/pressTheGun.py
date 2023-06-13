import threading

import cv2
import numpy as np
import winsound
from PIL import ImageGrab
from pynput import keyboard
from pynput import mouse

import G
from function.delay_ms import delay_ms, left_down, left_or_right_down, left_and_right_down
from function.segmentedMovement import _mouse
from PIL import ImageGrab, Image

flag_lock_obj = True  # 是否开启锁定的功能

flag = False

offset_x = 0

offset_y = 0

pressure_gun_switch = True

active_weapon = None

mouse_lock_open = mouse.Button.x1  # 开启锁人
mouse_lock_close = mouse.Button.x2  # 关闭锁人


def _watch_gun(_tC):
    global active_weapon
    while True:
        _gun = _tC.get()
        # print(_gun, '推流获取的')
        if not _gun:
            continue
        if _gun != active_weapon:
            active_weapon = _gun
        delay_ms(100)


def capture_screen_save(x, y, w, h):
    screen = ImageGrab.grab(bbox=[x, y, x + w, y + h])
    screen = np.array(screen)
    pil_image = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    pil_image.save("function/image/gun.png")
    return cv2.cvtColor(screen, cv2.COLOR_BGR2RGB), screen


def threadInitialization(_tC):
    # 在这里启动键盘监听和任务处理线程
    keyboard.Listener(on_release=on_release).start()
    threading.Thread(target=_watch_gun, args=(_tC,)).start()


def on_release(key):
    global pressure_gun_switch, flag
    if key == keyboard.Key.f5:
        capture_screen_save(G.left, G.top, G.width, G.height)
        print('截图成功!')
    if key == keyboard.Key.f11:
        pressure_gun_switch = not pressure_gun_switch
        if pressure_gun_switch:
            winsound.PlaySound('function/music/8855.wav', flags=1)
        else:
            winsound.PlaySound('function/music/close.wav', flags=1)
        print(f'识别压枪：{"开" if pressure_gun_switch else "关"}')


def down_gun_fun_c(modifier_value, t_C, no_wait_Queue):
    global flag, pressure_gun_switch, offset_x, offset_y, active_weapon
    print(f"识别压枪 load successful  自动计算压枪系数{modifier_value}   F11开关")
    print(f'识别压枪：{"开" if pressure_gun_switch else "关"}')
    threadInitialization(t_C)  # 进程加载
    while True:
        if left_and_right_down() and pressure_gun_switch:
            try:
                # 尝试按照后坐力模式压枪
                for i in range(len(G.recoil_patterns[active_weapon])):
                    if not left_or_right_down():
                        continue
                    offset_x = round(G.recoil_patterns[active_weapon][i][0] * modifier_value)
                    offset_y = round(G.recoil_patterns[active_weapon][i][1] * modifier_value)

                    # 非阻塞方式获取队列中的变量值 如果当前主线程正在自瞄x轴 则不需要压枪 否则 x轴也要压枪
                    if not no_wait_Queue.empty():
                        _mouse(0, offset_y)
                    else:
                        _mouse(offset_x, offset_y)
                    # no_wait_Queue.put((offset_x, offset_y)) # 通知主进程
                    delay_ms(int(G.recoil_patterns[active_weapon][i][2] * 1000))
                    flag = True  # 如果代码执行到这里，说明压枪成功，将标志设置为 True
                # delay_ms(5)
            except KeyError:
                flag = False
                pass  # 如果找不到后坐力模式，则不做任何操作
        else:
            flag = False
