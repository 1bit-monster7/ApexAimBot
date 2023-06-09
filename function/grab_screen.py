"""
@Time：  2023/1/12/012 20:11:37

grab_screen_win32_v2    640: 348        2560: 72
grab_screen_pyqt_v2     640: 175        2560: 42
g.cap()                 640: 77         2560: 34
win32_cls.capture()     640: 312        2560: 35
dxgi.cap()              640: 312        2560: 52
mss.mss().grab()        640: 74         2560: 36
"""
import numpy as np
import win32con
import win32gui
import win32ui

hwnd_title = dict()

is_show_not_find_window = True

window_size = None


def get_all_hwnd(hwnd, mouse):
    if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
        hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})


def update_hwnd_title():
    win32gui.EnumWindows(get_all_hwnd, 0)
    for h, t in hwnd_title.items():
        if t != "":
            # print(h, t)
            pass  # 获取所有进程的名称 用于得到例如Apex 的类名窗口截图


def grab_screen_win32_v2(window_title, grab_rect=None):
    global window_size
    hwnd = win32gui.FindWindow(None, window_title)
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    if grab_rect is None:
        if window_size is None or window_size == (0, 0):  # 第一次或者窗口大小变化了
            w = mfcDC.GetDeviceCaps(8)
            h = mfcDC.GetDeviceCaps(10)
            window_size = (w, h)
        else:
            w, h = window_size
    else:
        x, y, w, h = grab_rect

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)

    saveDC.BitBlt((0, 0), (w, h), mfcDC, (x, y), win32con.SRCCOPY)

    signed_ints_array = saveBitMap.GetBitmapBits(True)
    img = np.frombuffer(signed_ints_array, dtype="uint8")
    img.shape = (h, w, 4)
    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)  # 释放 DC 资源

    return img


def grab_gpt(window_title, grab_rect=None):
    hwnd = win32gui.FindWindow(None, window_title)
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    x, y, w, h = grab_rect
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)

    saveDC.BitBlt((0, 0), (w, h), mfcDC, (x, y), win32con.SRCCOPY)

    signed_ints_array = saveBitMap.GetBitmapBits(True)
    img = np.frombuffer(signed_ints_array, dtype="uint8")
    img.shape = (h, w, 4)
    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)  # 释放 DC 资源

    return img


class win32_cap:
    hwnd = None
    x, y, w, h = None, None, None, None

    def __init__(self):
        self.cDC = None
        self.dcObj = None
        self.wDC = None

    def Init(self, hwnd, left, top, width, height):
        self.hwnd = hwnd
        self.w = width
        self.h = height
        self.x = left
        self.y = top
        self.wDC = win32gui.GetWindowDC(self.hwnd)
        self.dcObj = win32ui.CreateDCFromHandle(self.wDC)
        self.cDC = self.dcObj.CreateCompatibleDC()

    def InitEx(self, hwnd, x, y, w, h):
        self.hwnd = hwnd
        self.x, self.y, self.w, self.h = x, y, w, h
        self.wDC = win32gui.GetWindowDC(self.hwnd)
        self.dcObj = win32ui.CreateDCFromHandle(self.wDC)
        self.cDC = self.dcObj.CreateCompatibleDC()

    def capture(self):
        try:
            dataBitMap = win32ui.CreateBitmap()
            dataBitMap.CreateCompatibleBitmap(self.dcObj, self.w, self.h)
            self.cDC.SelectObject(dataBitMap)
            self.cDC.BitBlt((0, 0), (self.w, self.h), self.dcObj, (self.x, self.y), 0x00CC0020)
            # 转换使得opencv可读
            signedIntsArray = dataBitMap.GetBitmapBits(True)
            cut_img = np.frombuffer(signedIntsArray, dtype='uint8')
            cut_img.shape = (self.h, self.w, 4)
            cut_img = cut_img[..., :3]  # 去除alpha
            win32gui.DeleteObject(dataBitMap.GetHandle())  # 释放资源
            cut_img = np.ascontiguousarray(cut_img)
            return cut_img
        except:
            print('error\n')
            return None

    def release_resource(self):
        win32gui.DeleteObject(self.wDC.GetHandle())
        self.wDC, self.dcObj, self.cDC = None, None, None
