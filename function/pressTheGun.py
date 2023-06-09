import threading

import win32api
import win32con
import winsound
from pynput import keyboard
from pynput import mouse

import G
from function.delay_ms import delay_ms
from function.segmentedMovement import _mouse

flag_lock_obj = True  # 是否开启锁定的功能

flag = False

offset_x = 0

offset_y = 0

pressure_gun_switch = True

active_weapon = None

mouse_lock_open = mouse.Button.x1  # 开启锁人
mouse_lock_close = mouse.Button.x2  # 关闭锁人


def is_key_pressed(key_code):
    """检查指定的按键是否被按下"""
    return win32api.GetAsyncKeyState(key_code) < 0


def is_left_click():
    return is_key_pressed(win32con.VK_LBUTTON)


def is_right_click():
    return is_key_pressed(win32con.VK_RBUTTON)


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


def threadInitialization(_tC):
    # 在这里启动键盘监听和任务处理线程
    keyboard.Listener(on_release=on_release).start()
    threading.Thread(target=_watch_gun, args=(_tC,)).start()


def on_release(key):
    global pressure_gun_switch, flag
    if key == keyboard.Key.f11:
        pressure_gun_switch = not pressure_gun_switch
        if pressure_gun_switch:
            winsound.PlaySound('function/music/8855.wav', flags=1)
        else:
            winsound.PlaySound('function/music/close.wav', flags=1)
        print(f'识别压枪：{"开" if pressure_gun_switch else "关"}')


def _down_gun_fun(modifier_value, t_C, no_wait_Queue):
    global flag, pressure_gun_switch, offset_x, offset_y, active_weapon
    print(f"识别压枪 load successful  自动计算压枪系数{ modifier_value}   F11开关")
    print(f'识别压枪：{"开" if pressure_gun_switch else "关"}')
    threadInitialization(t_C)  # 进程加载
    while True:
        if is_left_click() and pressure_gun_switch:
            try:
                # 尝试按照后坐力模式压枪
                for i in range(len(G.recoil_patterns[active_weapon])):
                    if not is_left_click() or not is_right_click():
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

            except KeyError:
                flag = False
                pass  # 如果找不到后坐力模式，则不做任何操作
        else:
            flag = False
