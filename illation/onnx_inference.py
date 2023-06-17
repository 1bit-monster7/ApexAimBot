#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import re

import numpy as np
import onnxruntime
import sys
import torch
import cv2
import torchvision
import os

from loguru import logger
from illation.data_augment import preproc as preprocess
from illation.demo_utils import multiclass_nms, demo_postprocess
from illation.visualize import vis


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2, eps=1e-7):
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        agnostic=False,
                        max_det=300):
    bs = prediction.shape[0]  # batch size
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    merge = False  # use merge-NMS
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
    return output


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


class Annotator:
    def __init__(self, im, line_width=None):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        self.lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


class Colors:
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


sftrt = None
sftrt1 = None
sftrt2 = None


class YOLOX_ONNX:
    def __init__(self, model, zhixingdu=0.5, nmszhi=0.5, trt=False, xiancun=2, fp16=True, shebei=0):
        global sftrt, sftrt1, sftrt2
        self.model = model  # 模型
        self.input_shape = '320, 320'
        self.Yv = ''
        self.fp16 = fp16
        self.trt = trt
        self.xiancun = int(xiancun)
        self.zhixingdu = zhixingdu
        self.nmszhi = nmszhi
        self.sffp16 = None
        self.shebei = shebei

        # 创建会话选项对象，并设置优化参数
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 0  # 设置并行线程数
        options.intra_op_num_threads = 0  # 设置并行线程数
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL  # 设置执行模式

        if False:
            from jiami import jiemi
            from aim_main import key_m
            key = key_m()

            # 定义获取文件内容函数
            def get_file_content(file_name):
                # 打开文件，读取内容，关闭文件
                f = open(file_name, "rb")
                content = f.read()
                f.close()
                # 返回文件内容
                return content

            cipher_text = get_file_content(self.model)
            # 对文件内容进行解密，得到一个jiemi类的实例
            model_onnx2 = jiemi(key, cipher_text)

            # 调用jiemi类的main方法，得到解密后的bytes格式数据
            plain_text = model_onnx2.main()

            if self.trt == True:
                try:
                    trt_xiancun = int(self.xiancun * 1024 * 1024 * 1024)
                    model_name = os.path.split(self.model)[-1]  # 这里会得到 'your_yolox300_320.onnx'
                    model_name = os.path.splitext(model_name)[0]  # 这里会去掉 '.onnx' 后缀
                    # # 定义当前文件夹路径
                    current_path = os.path.dirname(os.path.abspath(__file__))
                    # 定义weights文件夹的路径
                    weights_path = os.path.join(current_path, "weights")
                    # 如果 weights 文件夹不存在，就创建一个
                    if not os.path.exists(weights_path):
                        os.mkdir(weights_path)

                    trt_providers = [
                        ('TensorrtExecutionProvider', {
                            'device_id': self.shebei,
                            'trt_max_workspace_size': trt_xiancun,
                            'trt_fp16_enable': fp16,
                            'trt_engine_cache_enable': True,
                            'trt_engine_cache_path': f'{weights_path}/{model_name}',
                        }),
                        ('CUDAExecutionProvider', {
                            'device_id': self.shebei,
                            'arena_extend_strategy': 'kNextPowerOfTwo',
                            'gpu_mem_limit': self.xiancun * 1024 * 1024 * 1024,
                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                            'do_copy_in_default_stream': True,
                        })
                    ]
                    if sftrt != True:
                        self.session = onnxruntime.InferenceSession(plain_text, providers=trt_providers, sess_options=options)
                        sftrt2 = self.model
                        sftrt1 = self.session
                        sftrt = True
                    else:
                        if sftrt2 != self.model:
                            logger.error('检测到模型已更换，即将重新加载软件...')
                            # 用当前的Python解释器来执行一个新的程序，并替换当前的进程
                            os.execl(sys.executable, sys.executable, *sys.argv)
                        else:
                            self.session = sftrt1
                    logger.info(f'推理类型-ONNX-TRT推理')
                except:
                    try:
                        self.session = onnxruntime.InferenceSession(plain_text, providers=['CUDAExecutionProvider'], sess_options=options)
                        logger.info(f'推理类型-ONNX-CUDA推理')
                    except:
                        self.session = onnxruntime.InferenceSession(plain_text, providers=['CPUExecutionProvider'],
                                                                    sess_options=options)
                        logger.info(f'推理类型-ONNX-CPU推理')
            else:
                try:
                    self.session = onnxruntime.InferenceSession(plain_text, providers=['CUDAExecutionProvider'],
                                                                sess_options=options)
                    logger.info(f'推理类型-ONNX-CUDA推理')
                except:
                    self.session = onnxruntime.InferenceSession(plain_text, providers=['CPUExecutionProvider'],
                                                                sess_options=options)
                    logger.info(f'推理类型-ONNX-CPU推理')
        else:
            if self.trt == True:
                try:
                    trt_xiancun = int(self.xiancun * 1024 * 1024 * 1024)
                    model_name = os.path.split(self.model)[-1]  # 这里会得到 'your_yolox300_320.onnx'
                    model_name = os.path.splitext(model_name)[0]  # 这里会去掉 '.onnx' 后缀
                    # # 定义当前文件夹路径
                    current_path = os.path.dirname(os.path.abspath(__file__))
                    # 定义weights文件夹的路径
                    weights_path = os.path.join(current_path, "weights")
                    # 如果 weights 文件夹不存在，就创建一个
                    if not os.path.exists(weights_path):
                        os.mkdir(weights_path)

                    trt_providers = [
                        ('TensorrtExecutionProvider', {
                            'device_id': 0,
                            'trt_max_workspace_size': trt_xiancun,
                            'trt_fp16_enable': fp16,
                            'trt_engine_cache_enable': True,
                            'trt_engine_cache_path': f'{weights_path}/{model_name}',
                        }),
                        ('CUDAExecutionProvider', {
                            'device_id': self.shebei,
                            'arena_extend_strategy': 'kNextPowerOfTwo',
                            'gpu_mem_limit': self.xiancun * 1024 * 1024 * 1024,
                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                            'do_copy_in_default_stream': True,
                        })
                    ]
                    if sftrt != True:
                        logger.info(f'TRT模型加载中---首次可能会比较慢---注意事项(更换模型后会自动重启软件)')
                        self.session = onnxruntime.InferenceSession(self.model, providers=trt_providers, sess_options=options)
                        sftrt2 = self.model
                        sftrt1 = self.session
                        sftrt = True
                    else:
                        if sftrt2 != self.model:
                            logger.error('检测到模型已更换，即将重新加载软件...')
                            # 用当前的Python解释器来执行一个新的程序，并替换当前的进程
                            os.execl(sys.executable, sys.executable, *sys.argv)
                        else:
                            self.session = sftrt1
                    logger.info(f'推理类型-ONNX-TRT推理')
                except:
                    try:
                        self.session = onnxruntime.InferenceSession(self.model, providers=['CUDAExecutionProvider'], sess_options=options)
                        logger.info(f'推理类型-ONNX-CUDA推理')
                    except:
                        self.session = onnxruntime.InferenceSession(self.model, providers=['CPUExecutionProvider'],
                                                                    sess_options=options)
                        logger.info(f'推理类型-ONNX-CPU推理')
            else:
                try:
                    self.session = onnxruntime.InferenceSession(self.model, providers=['CUDAExecutionProvider'],
                                                                sess_options=options)
                    logger.info(f'推理类型-ONNX-CUDA推理')
                except:
                    self.session = onnxruntime.InferenceSession(self.model, providers=['CPUExecutionProvider'],
                                                                sess_options=options)
                    logger.info(f'推理类型-ONNX-CPU推理')

    def main(self, img1, xianshi=True):
        if self.Yv == 'YOLOX':
            input_shape = tuple(map(int, self.input_shape.split(',')))
            img, ratio = preprocess(img1, input_shape)

            ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
            output = self.session.run(None, ort_inputs)
            predictions = demo_postprocess(output[0], input_shape)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.nmszhi, score_thr=self.zhixingdu)
            if xianshi == True:
                if dets is not None:
                    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                    vis(img1, final_boxes, final_scores, final_cls_inds, conf=self.zhixingdu)
            return dets
        elif self.Yv == 'YOLOV5':
            try:
                colors = Colors()  # create instance for 'from utils.plots import colors'
                input_shape = tuple(map(int, self.input_shape.split(',')))
                img = letterbox(img1, (input_shape[0], input_shape[1]), stride=32,
                                auto=False)  # only pt use auto=True, but we are onnx
                img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                img = np.ascontiguousarray(img)
                img1 = np.ascontiguousarray(img1)
                im = torch.from_numpy(img).to(torch.device('cpu'))
                if self.sffp16 == True:
                    im = im.half()  # 把张量转换为float16类型
                else:
                    im = im.float()  # 把张量转换为float32类型
                im /= 255.0  # 0 - 255 to 0.0 - 1.0 # 把张量归一化到0-1之间，注意要用浮点数除法
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                im = im.cpu().numpy()  # torch to numpy
                y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[
                    0]  # inference onnx model to get the total output
                # non_max_suppression to remove redundant boxes
                y = torch.from_numpy(y).to(torch.device('cpu'))
                pred = non_max_suppression(y, conf_thres=self.zhixingdu, iou_thres=self.nmszhi, agnostic=False, max_det=1000)
                # print(pred)
                # transform coordinate to original picutre size
                for i, det in enumerate(pred):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img1.shape).round()
                if xianshi == True:
                    # initialize annotator
                    annotator = Annotator(img1, line_width=1)  # 创建一个注释器类的实例，用于在图片上画框和标签
                    # annotate the image
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class # 把类别转换为整数
                        annotator.box_label(xyxy, color=colors(c, True))  # 调用注释器类的方法，在图片上画框和标签，
                return det
            except onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument as e:
                # use regular expression to find the expected size
                match = re.search(r"Expected: (\d+)", str(e))
                pattern = r"expected: \(tensor\(float\)\)"  # 使用反斜杠转义括号
                string = "expected: (tensor(float))"
                match2 = re.search(pattern, string)
                if match:
                    expected_size = int(match.group(1))  # convert the matched string to integer
                    self.input_shape = f"{expected_size}, {expected_size}"
                    logger.error(f'已修改当前模型尺寸为：{self.input_shape}')
                elif match2:
                    self.sffp16 = False
                    logger.error(f'已修改当前模型精度为：{self.sffp16}')
                else:
                    logger.error(f'错误错误: {str(e)}')
        else:
            try:

                input_shape = tuple(map(int, self.input_shape.split(',')))
                img, ratio = preprocess(img1, input_shape)

                ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
                output = self.session.run(None, ort_inputs)
                predictions = demo_postprocess(output[0], input_shape)[0]

                boxes = predictions[:, :4]
                scores = predictions[:, 4:5] * predictions[:, 5:]

                boxes_xyxy = np.ones_like(boxes)
                boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
                boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
                boxes_xyxy /= ratio
                dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.nmszhi, score_thr=self.zhixingdu)
                self.Yv = 'YOLOX'
                if xianshi == True:
                    if dets is not None:
                        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                        vis(img1, final_boxes, final_scores, final_cls_inds, conf=self.zhixingdu)
                return dets
            except:
                try:
                    colors = Colors()  # create instance for 'from utils.plots import colors'
                    input_shape = tuple(map(int, self.input_shape.split(',')))
                    img = letterbox(img1, (input_shape[0], input_shape[1]), stride=32,
                                    auto=False)  # only pt use auto=True, but we are onnx
                    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    img = np.ascontiguousarray(img)
                    img1 = np.ascontiguousarray(img1)
                    im = torch.from_numpy(img).to(torch.device('cpu'))
                    if self.sffp16 == True:
                        im = im.half()  # 把张量转换为float16类型
                    else:
                        im = im.float()  # 把张量转换为float32类型
                    im /= 255.0  # 0 - 255 to 0.0 - 1.0 # 把张量归一化到0-1之间，注意要用浮点数除法
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    im = im.cpu().numpy()  # torch to numpy
                    y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[
                        0]  # inference onnx model to get the total output
                    # non_max_suppression to remove redundant boxes
                    y = torch.from_numpy(y).to(torch.device('cpu'))
                    pred = non_max_suppression(y, conf_thres=self.zhixingdu, iou_thres=self.nmszhi, agnostic=False, max_det=1000)
                    # transform coordinate to original picutre size
                    self.Yv = 'YOLOV5'
                    for i, det in enumerate(pred):
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img1.shape).round()
                    if xianshi == True:
                        # initialize annotator
                        annotator = Annotator(img1, line_width=1)  # 创建一个注释器类的实例，用于在图片上画框和标签
                        # annotate the image
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class # 把类别转换为整数
                            annotator.box_label(xyxy, color=colors(c, True))  # 调用注释器类的方法，在图片上画框和标签，
                    return det
                except onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument as e:
                    # use regular expression to find the expected size
                    match = re.search(r"Expected: (\d+)", str(e))
                    pattern = r"expected: \(tensor\(float\)\)"  # 使用反斜杠转义括号
                    string = "expected: (tensor(float))"
                    match2 = re.search(pattern, string)
                    if match:
                        expected_size = int(match.group(1))  # convert the matched string to integer
                        self.input_shape = f"{expected_size}, {expected_size}"
                        logger.error(f'已修改当前模型尺寸为：{self.input_shape}')
                    elif match2:
                        self.sffp16 = False
                        logger.error(f'已修改当前模型精度为：{self.sffp16}')
                    else:
                        logger.error(f'错误错误: {str(e)}')
