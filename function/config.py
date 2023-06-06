
import win32con
from pynput import mouse

mouse_lock_open = mouse.Button.x1  # 开启锁人
mouse_lock_close = mouse.Button.x2  # 关闭锁人

# 定义瞄准模式对应的按键 1 左 2右 3左右
AIM_MODE_KEYS = {
    1: win32con.VK_LBUTTON,
    2: win32con.VK_RBUTTON,
    3: (win32con.VK_RBUTTON, win32con.VK_LBUTTON),
    4: win32con.VK_XBUTTON2,
}

grab_window_title = 'Apex Legends'

smoothness_x = 0.5  # 平滑度
smoothness_y = 0.6  # Y轴 0 - 1 # 越大越快
smoothness = 0.8  # PID 平滑度 0 - 1  #越大 越小 PID
