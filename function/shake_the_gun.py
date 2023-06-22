from ctypes import windll

import win32api
import win32con
import winsound
from pynput import keyboard

from function.delay_ms import delay_ms
from function.logitech import Logitech

timeBeginPeriod = windll.winmm.timeBeginPeriod
timeEndPeriod = windll.winmm.timeEndPeriod
Sleep = windll.kernel32.Sleep

down = True
R = 1
E = 2
T = 7


def shake_the_gun_fun_gpt():
    while True:
        # 左右键按下时 且 压枪控制打开时
        if down and win32api.GetAsyncKeyState(win32con.VK_LBUTTON) < 0:
            delay_move(R, R, T)  # 移动鼠标
            delay_move(-R, -R, T)
            delay_move(0, E, T)
            delay_move(0, -E, T)


def delay_move(x, y, ms=0):
    Logitech.mouse.move(x, y)
    delay_ms(ms)


def on_press(key):
    global down
    if key == keyboard.Key.f11:
        down = not down
        if down:
            winsound.PlaySound('otherFeatures/music/8855.wav', flags=1)
        else:
            winsound.PlaySound('otherFeatures/music/close.wav', flags=1)
        print(f'抖枪：{"开" if down else "关"}')


def shake_gan_main():
    print(f"抖枪宏 load successful 抖动幅度{R} 抖动延时{T}ms HOME开关 {'开' if down else '关'}")
    listener_keyboard = keyboard.Listener(on_press=on_press)
    listener_keyboard.start()
    shake_the_gun_fun_gpt()
