import multiprocessing
import os
import platform
import sys
import threading
from multiprocessing import set_start_method, Queue, Process, freeze_support
from pathlib import Path
from timeit import default_timer as timer

import cv2
import numpy as np
import torch
import win32api
import win32con
import winsound
from pynput import mouse

from G import params_list
from function.OB import Publisher_Ui, Subscriber_Fun
from function.PID_INCREMENT import PID_PLUS_GPT
from function.automatic_armor_change import automatic_armor_change_func
from function.configUtils import get_ini, _set_config
from function.delay_ms import delay_ms
from function.grab_screen import update_hwnd_title, grab_gpt
from function.identify_firearms import find_gun_main
from function.logitech import Logitech
from function.pressTheGun import _down_gun_fun
from function.segmentedMovement import segmented_movement_xy
from function.shake_the_gun import shake_gan_main
from function.web_ui import create_ui
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device

# 取当前py文件运行目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 管理所有控制参数 同步ini
min_step = 3
max_step = 6
grab_window_title = 'Apex Legends'
screen_width = 1920
screen_height = 1080
grab_rectangle = 0
debug = 0
is_show_top_window = 0
aim_mod = 0
conf_thres = 0.5
iou_thres = 0.01
weight = 'APEX416.engine'
model_imgsz = 416
grab_width = 600
grab_height = 300
pid_x_p = 0.7
pid_x_i = 0.045
pid_x_d = 0.002
pid_y_p = 0.8
pid_y_i = 0
pid_y_d = 0
modifier_value = 0.88
sens = 5  # 鼠标灵敏度
ads = 1  # 开镜灵敏度
shake_coefficient = 2.290904
shake_coefficient_y = 2
shake_delay = 6
# 静态参数
name_list = None
top_window_name = 'win'
is_lock_radio = True
instantiation_pid_x = None
instantiation_pid_y = None
mouse_lock_open = mouse.Button.x1  # 开启锁人
mouse_lock_close = mouse.Button.x2  # 关闭锁人
top_window_width = 200
flag = False
pressure_gun_switch = True
active_weapon = 'R-99'
model = None
# 发布者 用于通知所有订阅者
BIT_GOD = None
is_loading = False
# 订阅者 收到订阅执行相关命令
WATCH_PERSON = None
# 第一次加载程序
first_load = 0

# 通信 Queue
shake_r = 0.0
shake_t = 0.0
md_value = 0.0


def is_key_pressed(key_code):
    """检查指定的按键是否被按下"""
    return win32api.GetAsyncKeyState(key_code) < 0


def is_left_click():
    return is_key_pressed(win32con.VK_LBUTTON)


def is_right_click():
    return is_key_pressed(win32con.VK_RBUTTON)


def _init_main(msg=''):
    global model, name_list, is_loading, modifier_value, sens, ads, first_load, shake_r, shake_t, shake_y
    print(msg)
    print(msg)
    print(msg)
    global instantiation_pid_x, instantiation_pid_y
    # 同步所有ini的参数到py
    for key in params_list:
        globals()[key] = get_ini(key)
    # pid 初始化
    instantiation_pid_x = PID_PLUS_GPT(0, pid_x_p, pid_x_i, pid_x_d)
    instantiation_pid_y = PID_PLUS_GPT(0, pid_y_p, pid_y_i, pid_y_d)

    # 压枪系数计算
    zoom_sens = 1 / ads
    modifier_value = 4 / sens * zoom_sens

    # 更新进程共享变量
    shake_r.value, shake_y.value, shake_t.value = shake_coefficient, shake_coefficient_y, shake_delay  # 共享变量赋值
    _set_config('modifier_value', modifier_value)  # 计算后的值存储

    # 第一次加载或debu模式为真
    if debug or first_load == 0:
        first_load += 1
        #  1. 模型加载
        data = ROOT / 'function' / 'yaml' / 'coco128.yaml'
        device = select_device('')
        w = ROOT / 'function' / 'weights' / weight
        model = DetectMultiBackend(w, device=device, dnn=False, data=data, fp16=True)
        model.warmup(imgsz=(1, 3, *[model_imgsz, model_imgsz]))  # warmup
        name_list = [name for name in model.names.values()]  # 拿到标签类
        print(name_list, '当前模型分类')
        print(f"分辨率:{screen_width}*{screen_height}p 截图宽高:{grab_width, grab_height} 窗口标题:{grab_window_title} 窗口开关：{is_show_top_window}")
    is_loading = False  # 加载完成可以开始推理


def send_nearest_pos_to_mouse_ctrl(box_list):
    global name_list
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

    y_offset = half_grab_height * 0.05

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
    else:
        return min_position, box_width, box_height


def delay_move(x, y, ms=0):
    Logitech.mouse.move(x, y)
    delay_ms(ms)


def _s(num):
    return "{:.2f}".format(num)


def on_click(x, y, button, pressed):
    global is_lock_radio
    if not pressed:
        if button == mouse_lock_open:
            is_lock_radio = True
            winsound.PlaySound('function/music/ok.wav', flags=1)
            print("锁定功能：开")
        if button == mouse_lock_close:
            winsound.PlaySound('function/music/close.wav', flags=1)
            is_lock_radio = False
            print("锁定功能：关")


def processInitialization(no_wait_Queue):
    _C = Queue()

    Process(target=automatic_armor_change_func).start()  # 自动换甲
    Process(target=find_gun_main, args=(_C,)).start()  # 自动识别压枪
    Process(target=_down_gun_fun, args=(modifier_value, _C, no_wait_Queue)).start()  # 得到压枪数据
    Process(target=shake_gan_main, args=(shake_r, shake_t, shake_y)).start()  # 抖枪宏 通用宏


def threadInitialization(no_wait_Queue):
    update_hwnd_title()  # get class windows
    mouse.Listener(on_click=on_click).start()
    threading.Thread(target=run_ai, args=(no_wait_Queue,)).start()  # ai run


def interface_img(img):
    global model_imgsz, conf_thres, iou_thres, model
    stride, names = model.stride, model.names

    img = cv2.resize(img, (model_imgsz, model_imgsz))

    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # 转换

    # Load image
    im = letterbox(img, model_imgsz, stride=stride, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    # GPU optimization
    torch.backends.cudnn.benchmark = True

    with torch.no_grad():  # Disable gradient calculation
        im = torch.from_numpy(im).to(model.device)
        im = im.bfloat16() if model.fp16 else im.float()  # uint8 to bfloat16/float32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im.unsqueeze(0)  # expand for batch dim

        # Inference
        pred = model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)

    box_list = []
    gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    # Define CUDA stream for synchronization
    cuda_stream = torch.cuda.Stream()

    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                className = names[int(cls)]

                if className == "teammate":  # Filter teammates
                    continue

                conf_str = f"{int(100 * float(conf))}"

                box_list.append((className, *xywh, conf_str))

    with torch.cuda.stream(cuda_stream):
        torch.cuda.synchronize()

    return box_list


def run_ai(no_wait_Queue):
    global model
    fps_count = 0  # fps计次
    fps_list = []  # fps list
    while True:
        if is_loading:
            continue
        else:
            # 非阻塞方式获取队列中的变量值
            # if not no_wait_Queue.empty():
            #     offset_x, offset_y = no_wait_Queue.get_nowait()
            # else:
            #     offset_x, offset_y = 0, 0
            # 2. 截图
            fps_tag = timer()
            rect = (int(screen_width / 2 - grab_width / 2), int(screen_height / 2 - grab_height / 2), grab_width, grab_height)  # left,top,width,height
            img = grab_gpt(window_title=grab_window_title, grab_rect=rect)
            _search_time = _s((timer() - fps_tag) * 1000)  # 截图耗时
            # 3. 推理
            t1 = timer()
            box_list = interface_img(img)
            _pred_time = _s((timer() - t1) * 1000)  # 推理耗时
            # FPS 计算
            _for_time = _s((timer() - fps_tag) * 1000)
            _fps = (1 / float(_for_time)) * 1000
            fps_list.append(_fps)
            fps_count += 1

            if fps_count % 500 == 0:
                average_fps = np.average(fps_list)
                fps_list = []
                fps_count = 0
                print(f"500抡平均FPS：{_s(average_fps)}")

            # 4. 发送最近坐标
            result = send_nearest_pos_to_mouse_ctrl(box_list)

            # 有数据则锁人
            if result:
                pos_min, box_width, box_height = result

                abs_x = pos_min[0] if pos_min[0] > 0 else -pos_min[0]

                abs_y = pos_min[1] if pos_min[1] > 0 else -pos_min[1]

                _range = 1  # 0.5 - 1  值越小范围越小 越大距离越远越锁
                _range_y = 1.2
                # have_luck = abs_x <= (box_width * _range) and abs_y <= (box_height * _range)
                have_luck = abs_x <= (box_width * _range) and abs_y <= (box_height * _range_y)

                if aim_mod == 0:
                    press = is_left_click()
                elif aim_mod == 1:

                    press = is_right_click()
                elif aim_mod == 2:
                    press = is_left_click() or is_right_click()
                else:
                    return print('瞄准模式异常')

                if is_lock_radio and press and have_luck:
                    no_wait_Queue.put(True)  # 告诉压枪进程 不要压x轴
                    # 第一次时不启用i 解决过冲问题
                    _pid_x = int(instantiation_pid_x.getMove(int(pos_min[0])))
                    segmented_movement_xy(_pid_x, 0, min_step, max_step)  # 分段移动


def main():
    global BIT_GOD, WATCH_PERSON, shake_r, shake_t
    # 注册一个发布者
    BIT_GOD = Publisher_Ui()
    # 注册一个订阅者
    WATCH_PERSON = Subscriber_Fun(_init_main)
    # 将订阅者注册到发布者中
    BIT_GOD.add_subscriber(WATCH_PERSON)
    # 创建队列，用于传递变量值
    no_wait_Queue = multiprocessing.Queue()

    processInitialization(no_wait_Queue)  # 进程初始化
    threadInitialization(no_wait_Queue)  # 线程初始化


if __name__ == '__main__':
    freeze_support()  # 解决多线程 pyinstall打包bug
    set_start_method('spawn')  # 多进程上下文设置

    # 初始化之前创建进程共享变量
    manager = multiprocessing.Manager()
    shake_r = manager.Value('d', 0.0)
    shake_t = manager.Value('d', 0.0)
    shake_y = manager.Value('d', 0.0)

    _init_main()  # 初始化参数
    main()  # 主程序
    create_ui(BIT_GOD)  # 创建ui 并将发布者传递给ui
