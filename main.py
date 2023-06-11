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
import win32con
import win32gui
import winsound
from pynput import mouse

from G import params_list
from function.OB import Publisher_Ui, Subscriber_Fun
from function.PID_INCREMENT import PID_PLUS_PLUS
from function.automatic_armor_change import automatic_armor_change_func
from function.configUtils import get_ini, set_config
from function.delay_ms import delay_ms, left_down_not_right, left_down, right_down, left_or_right_down
from function.grab_screen import update_hwnd_title, grab_gpt
from function.identify_firearms import find_gun_main
from function.logitech import Logitech
from function.pressTheGun import down_gun_fun_c
from function.segmentedMovement import generate_random_int, _mouse
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
    instantiation_pid_x = PID_PLUS_PLUS(0, pid_x_p, pid_x_i, pid_x_d)
    instantiation_pid_y = PID_PLUS_PLUS(0, pid_y_p, pid_y_i, pid_y_d)

    # 压枪系数计算
    zoom_sens = 1 / ads
    modifier_value = 4 / sens * zoom_sens

    # 更新进程共享变量
    shake_r.value, shake_y.value, shake_t.value = shake_coefficient, shake_coefficient_y, shake_delay  # 共享变量赋值
    set_config('modifier_value', modifier_value)  # 计算后的值存储

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
        print(f"分辨率:{screen_width}*{screen_height}p 截图宽高:{grab_width, grab_height} 窗口标题:{grab_window_title} 窗口开关：{is_show_top_window}  ")
    is_loading = False  # 加载完成可以开始推理


def send_nearest_pos_to_mouse_ctrl(box_list):
    global name_list
    if not box_list:
        return None

    min_distance_sq = float('inf')
    min_position = None

    box_width = None
    box_height = None

    half_grab_width = grab_width / 2
    half_grab_height = grab_height / 2

    for box in box_list:
        if box[0] not in name_list:
            continue
        box_center_x = box[1] * grab_width
        box_center_y = box[2] * grab_height
        box_width = box[3] * grab_width  # 目标框宽度
        box_height = box[4] * grab_height  # 目标框高度

        distance_sq = (box_center_x - half_grab_width) ** 2 + (box_center_y - half_grab_height) ** 2
        if distance_sq < min_distance_sq:
            min_distance_sq = distance_sq
            min_position = (box_center_x - half_grab_width, box_center_y - half_grab_height)

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
    Process(target=down_gun_fun_c, args=(modifier_value, _C, no_wait_Queue)).start()  # 得到压枪数据
    Process(target=shake_gan_main, args=(shake_r, shake_t, shake_y)).start()  # 抖枪宏 通用宏


def threadInitialization(no_wait_Queue):
    update_hwnd_title()  # get class windows
    mouse.Listener(on_click=on_click).start()
    threading.Thread(target=run_ai, args=(no_wait_Queue,)).start()  # ai run


def interface_img_gpt(img):
    global model_imgsz, conf_thres, iou_thres, model
    stride, names = model.stride, model.names,

    # img = cv2.resize(img, (model_imgsz, model_imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    im = letterbox(img, model_imgsz, stride=stride, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)

    torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        im = torch.from_numpy(im).to(model.device)
        im = im.bfloat16() if model.fp16 else im.float()
        im /= 255

        if len(im.shape) == 3:
            im = im.unsqueeze(0)

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=5)

    box_list = []
    gn = torch.tensor(img.shape)[[1, 0, 1, 0]]

    cuda_stream = torch.cuda.Stream()

    if len(pred[0]):
        det = pred[0]
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()

        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

            className = names[int(cls)]

            if className == "teammate":
                continue

            conf_str = f"{int(100 * float(conf))}"

            box_list.append((className, *xywh, conf_str))

    with torch.cuda.stream(cuda_stream):
        torch.cuda.synchronize()

    return box_list


def draw_fps(img, fps_tag, fps_list):
    t = timer() - fps_tag
    if len(fps_list) > 10:
        fps_list.pop(0)
        fps_list.append(t)
    else:
        fps_list.append(t)
    cv2.putText(img, str(int(1 / np.mean(fps_list))), (20, 100), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255), 6)
    return img


def draw_box(img, box_list):
    global name_list
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


@torch.no_grad()  # 不要删 (do not delete it )
def run_ai(no_wait_Queue):
    global model
    fps_l = []
    rect = (int(screen_width / 2 - grab_width / 2), int(screen_height / 2 - grab_height / 2), grab_width, grab_height)
    while True:
        # 初始化
        _range = 1.2 if left_down_not_right() else 0.8  # 0.5 - 1  值越小范围越小 越大距离越远越锁
        _range_y = 0.8
        if is_loading:
            rect = (int(screen_width / 2 - grab_width / 2), int(screen_height / 2 - grab_height / 2), grab_width, grab_height)
            continue
        else:
            # 2. 截图
            fps_tag = timer()
            img = grab_gpt(window_title=grab_window_title, grab_rect=rect)
            _search_time = _s((timer() - fps_tag) * 1000)  # 截图耗时
            # 3. 推理
            t1 = timer()
            box_list = interface_img_gpt(img)
            _pred_time = _s((timer() - t1) * 1000)  # 推理耗时

            t2 = timer()
            # 4. 发送最近坐标
            result = send_nearest_pos_to_mouse_ctrl(box_list)
            _result_timer = _s((timer() - t2) * 1000)  # 推理耗时

            # FPS 计算
            _for_time = _s((timer() - fps_tag) * 1000)
            _fps = (1 / float(_for_time)) * 1000

            print(f"截图时间：{_search_time}  |  推理时间：{_pred_time}  |  处理坐标耗费时间：{_result_timer}  |  fps：{_fps}")

            # 有数据则锁人
            if result:
                pos_min, box_width, box_height = result

                abs_x = abs(pos_min[0])

                abs_y = abs(pos_min[1])

                if aim_mod == 0:
                    press = left_down()
                elif aim_mod == 1:
                    press = right_down()
                elif aim_mod == 2:
                    press = left_or_right_down()
                else:
                    return print('瞄准模式异常')
                have_luck = abs_x <= (box_width * _range) and abs_y <= (box_height * _range_y)

                if is_lock_radio and press and have_luck:
                    no_wait_Queue.put(True)  # 告诉压枪进程 不要压x轴
                    # random_step = generate_random_int(min_step, max_step)
                    if left_down_not_right():
                        _pid_x = int(instantiation_pid_x.getMove(pos_min[0], min_step))
                        _pid_y = int(instantiation_pid_y.getMove(pos_min[1] - (box_height * 0.2)))
                        _mouse(_pid_x, _pid_y)
                        print(f"移动距离: x：{_pid_x} y：{_pid_y} 最大限制步长:{min_step}")
                    else:
                        _pid_x = int(instantiation_pid_x.getMove(pos_min[0], max_step))
                        _pid_y = int(instantiation_pid_y.getMove(pos_min[1] - (box_height * 0.15)))
                        _mouse(_pid_x, _pid_y)
                        print(f"移动距离: x：{_pid_x} y：{_pid_y} 最大限制步长:{max_step}")
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            if is_show_top_window:
                # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
                # 5. 画框
                # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
                img = draw_box(img, box_list)

                # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
                # 6. FPS
                # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
                img = draw_fps(img, fps_tag, fps_l)

                # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
                # 7. 窗口显示
                # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
                cv2.namedWindow(top_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(top_window_name, top_window_width,
                                 int(top_window_width * grab_height / grab_width))  # 重置窗口大小
                hwnd = win32gui.FindWindow(None, top_window_name)
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 1900 - top_window_width, 150, 0, 0,
                                      win32con.SWP_NOSIZE)
                cv2.imshow(top_window_name, img)
                cv2.waitKey(1)


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
