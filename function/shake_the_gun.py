from ctypes import windll

import win32api
import winsound
from pynput import keyboard

from function.delay_ms import delay_ms
from function.logitech import Logitech

timeBeginPeriod = windll.winmm.timeBeginPeriod
timeEndPeriod = windll.winmm.timeEndPeriod
Sleep = windll.kernel32.Sleep

pressure_gun_switch = True

R = 3  # x抖动单位像素
E = 2  # y抖动单位像素
P = 0

left_clicked = False


def shake_the_gun_fun():
    while True:
        # 右键按下时 且 压枪控制打开时
        if pressure_gun_switch and right_click_state() and left_click_state:
            delay_move(-R, R, 0.01 * 1000)
            delay_move(R, R, 0.01 * 1000)
            delay_move(R, -R, 0.01 * 1000)
            delay_move(-R, -R, 0.01 * 1000)
            delay_move(0, E, 0.01 * 1000)


def _dt_gun():
    x_vibration_amplitude = 2  # 左右震动幅度，根据自己的灵敏度进行微调
    y_vibration_amplitude = 3  # 垂直震动幅度，根据自己的灵敏度进行微调
    vibration_speed = 4  # 震动速度，根据自己的灵敏度进行微调
    xWave = 0  # 总体向右偏移方向
    yWave = 3  # 总体向下偏移方向，根据自己的灵敏度进行微调
    while True:
        if left_click_state() and pressure_gun_switch and right_click_state():
            delay_ms(vibration_speed)
            Logitech.mouse.move(0, -y_vibration_amplitude)  # 向上
            delay_ms(vibration_speed)
            Logitech.mouse.move(x_vibration_amplitude, y_vibration_amplitude)  # 向右
            delay_ms(vibration_speed)
            Logitech.mouse.move(-x_vibration_amplitude, y_vibration_amplitude)  # 向下
            delay_ms(vibration_speed)
            Logitech.mouse.move(-x_vibration_amplitude, -y_vibration_amplitude)  # 向左
            delay_ms(vibration_speed)
            Logitech.mouse.move(x_vibration_amplitude, 0)  # 中心
            delay_ms(vibration_speed)
            Logitech.mouse.move(xWave, yWave)  # 总体偏移


def delay_move(x, y, ms=0):
    Logitech.mouse.move(x, y)
    delay_ms(ms)




def on_press(key):
    global pressure_gun_switch
    if key == keyboard.Key.f4:
        pressure_gun_switch = not pressure_gun_switch
        if pressure_gun_switch:
            winsound.PlaySound('function/music/8855.wav', flags=1)
        print(f'自动压枪：{"开" if pressure_gun_switch else "关"}')


def left_click_state():
    left_click = win32api.GetKeyState(0x01)
    return left_click < 0


def right_click_state():
    right_click = win32api.GetKeyState(0x02)
    return right_click < 0


def shake_gan_main():
    print('抖枪宏 load successful')
    print(f'抖枪宏：{"开" if pressure_gun_switch else "关"}')
    listener_keyboard = keyboard.Listener(on_press=on_press)
    listener_keyboard.start()
    # _dt_gun()
    shake_the_gun_fun()
