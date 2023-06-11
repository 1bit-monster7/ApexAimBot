from ctypes import windll

import winsound
from pynput import keyboard

from function.delay_ms import delay_ms, left_down
from function.logitech import Logitech

timeBeginPeriod = windll.winmm.timeBeginPeriod
timeEndPeriod = windll.winmm.timeEndPeriod
Sleep = windll.kernel32.Sleep

pressure_gun_switch = True


def shake_the_gun_fun(skr, skt, sky):
    while True:
        r = skr.value
        t = skt.value
        y = sky.value
        # 左右键按下时 且 压枪控制打开时
        if pressure_gun_switch and left_down():
            delay_move(-r, r, t)
            delay_move(r, r, t)
            delay_move(r, -r, t)
            delay_move(-r, -r, t)


def delay_move(x, y, ms=0):
    Logitech.mouse.move(x, y)
    delay_ms(ms)


def on_press(key):
    global pressure_gun_switch
    if key == keyboard.Key.f12:
        pressure_gun_switch = not pressure_gun_switch
        if pressure_gun_switch:
            winsound.PlaySound('function/music/8855.wav', flags=1)
        else:
            winsound.PlaySound('function/music/close.wav', flags=1)
        print(f'抖枪：{"开" if pressure_gun_switch else "关"}')


def shake_gan_main(shake_r, shake_t, shake_y):
    print(f"抖枪宏 load successful 抖动幅度{shake_r.value} 抖动延时{shake_t.value} F12开关")
    print(f'抖枪宏：{"开" if pressure_gun_switch else "关"}')
    listener_keyboard = keyboard.Listener(on_press=on_press)
    listener_keyboard.start()
    # _dt_gun()
    shake_the_gun_fun(shake_r, shake_t, shake_y)
