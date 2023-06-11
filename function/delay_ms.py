from ctypes import windll

import win32api
import win32con

timeBeginPeriod = windll.winmm.timeBeginPeriod
timeEndPeriod = windll.winmm.timeEndPeriod
Sleep = windll.kernel32.Sleep


def delay_ms(ms):
    timeBeginPeriod(1)
    Sleep(int(ms))
    timeEndPeriod(1)


def left_or_right_down():
    # 判断左右键同时按下
    if win32api.GetAsyncKeyState(win32con.VK_LBUTTON) < 0 or win32api.GetAsyncKeyState(win32con.VK_RBUTTON) < 0:
        # print('左键或者右键按下')
        return True
    else:
        return False


def left_and_right_down():
    # 判断左右键同时按下
    if win32api.GetAsyncKeyState(win32con.VK_LBUTTON) < 0 and win32api.GetAsyncKeyState(win32con.VK_RBUTTON) < 0:
        # print('左右键同时按下')
        return True
    else:
        return False


def left_down():
    # 判断按了左键
    if win32api.GetAsyncKeyState(win32con.VK_LBUTTON) < 0:
        # print('按了左键')
        return True
    else:
        return False


def right_down():
    # 判断只按了左键
    if win32api.GetAsyncKeyState(win32con.VK_LBUTTON) < 0:
        # print('按了左键')
        return True
    else:
        return False


def left_down_not_right():
    # 判断只按了左键，没按右键
    if win32api.GetAsyncKeyState(win32con.VK_LBUTTON) < 0 and win32api.GetAsyncKeyState(win32con.VK_RBUTTON) >= 0:
        # print('只按了左键')
        return True
    else:
        return False


def right_down_not_left():
    # 判断只按了右键，没按左键
    if win32api.GetAsyncKeyState(win32con.VK_LBUTTON) >= 0 and win32api.GetAsyncKeyState(win32con.VK_RBUTTON) < 0:
        # print('只按了右键')
        return True
    else:
        return False
