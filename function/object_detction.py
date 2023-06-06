import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from function.configUtils import get_config_from_key
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (non_max_suppression, scale_boxes, xyxy2xywh)
from utils.torch_utils import select_device

# 取当前py文件运行目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

_model_imgsz = [320, 320]
_conf_thres = None
_iou_thres = None
_weights = None


# 初始化模型参数
def _init_model():
    global _model_imgsz, _conf_thres, _iou_thres, _weights
    size = get_config_from_key('_model_imgsz')
    _model_imgsz = [size, size]
    _conf_thres = get_config_from_key('_conf_thres')
    _iou_thres = get_config_from_key('_iou_thres')
    fileName = get_config_from_key('_weight')
    _weights = ROOT / 'weights' / fileName  # 权重文件路径

    return [fileName, _model_imgsz, _conf_thres, _iou_thres, _weights]


def load_model():
    auth = _init_model()  # 初始化model参数
    global _model_imgsz, _weights
    # --- --- --- --- --- --- --- ---  --- --- --- --- --- --- --- --- #
    # 加载模型
    # --- --- --- --- --- --- --- ---  --- --- --- --- --- --- --- --- #
    data = ROOT / 'yaml' / 'coco128.yaml'
    device = select_device('')
    model = DetectMultiBackend(_weights, device=device, dnn=False, data=data, fp16=True)
    model.warmup(imgsz=(1, 3, *_model_imgsz))  # warmup
    name_list = [name for name in model.names.values()]  # 拿到标签类

    return model, name_list, auth


def interface_img(img, model):
    global _conf_thres, _iou_thres
    stride, names = model.stride, model.names

    img = cv2.resize(img, (_model_imgsz[0], _model_imgsz[1]))

    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # 转换

    # Load image
    im = letterbox(img, _model_imgsz[0], stride=stride, auto=True)[0]  # padded resize
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
        pred = non_max_suppression(pred, _conf_thres, _iou_thres, max_det=1000)

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
