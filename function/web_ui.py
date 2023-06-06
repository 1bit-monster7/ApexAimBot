import configparser
import os
import sys

import gradio as gr
from gradio.themes import Color

from function.configUtils import get_config_from_key, _set_config, _write

dark_themes = Color(
    name="dark_themes",
    c50="#000000",  # 黑 主色调
    c100="#ffffff",  # 白 副色调
    c200="#ffffff",
    c300="#ffffff",
    c400="#ffffff",
    c500="#303030",  # 文字颜色
    c600="#404258",  # radio 颜色
    c700="#B6EADA",
    c800="#ffffff",
    c900="#ffffff",
    c950='#ffffff',
)

config = configparser.ConfigParser()
# 从1bit.ai.config文件中读取参数和值
config.read("1bit.ai.config")

textGroup = 'group'  # 分组名称
# 定义一个公共字典对象 用于保存参数 这里的数组key顺序要和下面的传参顺序一致
arr = [
    '_debug',
    '_is_show_top_window',
    '_conf_thres',
    '_iou_thres',
    '_weight',
    '_model_imgsz',
    '_grab_width',
    '_grab_height',
    'pid_x_P',
    'pid_x_I',
    'pid_x_D',
    'pid_x_min',
    'pid_x_max',
    'pid_y_P',
    'pid_y_I',
    'pid_y_D',
    'pid_y_min',
    'pid_y_max',
]


def submit(*args):
    print(f"保存并拿到参数keys数组：{args}")
    count = 0
    for val in args:
        key = arr[count]  # key
        count += 1
        _set_config(key, val)
    _write()  # 写入ini


def list_weights_files():
    weights_folder = os.path.join(os.getcwd(), 'function', 'weights')
    files = os.listdir(weights_folder)
    weight_files = []
    for f in files:
        if f.endswith('.engine') or f.endswith('.pt') or f.endswith('.onnx'):
            weight_files.append(f)
    # print(weight_files, '当前项目所有权重文件')
    return weight_files


def restart_program():
    # 用当前的Python解释器来执行一个新的程序，并替换当前的进程
    os.execl(sys.executable, sys.executable, *sys.argv)


def create_ui():
    models_files_list = list_weights_files()  # 获取所有权重文件
    with gr.Blocks(
            css=".gradio-container {background-color: #03001C;max-width:100vw!important;margin:0!important;}",
            theme=gr.themes.Soft(primary_hue=dark_themes)) as demo:
        # with gr.Blocks(theme=gr.themes.default) as demo:
        with gr.Tab("APEX Ai Config"):
            with gr.Row():  # 并行显示，可开多列
                _debug = gr.Radio(['开启', '关闭'], label="Debug", info="实时调整参数", value=get_config_from_key('_debug'))
                _is_show_top_window = gr.Radio(['开启', '关闭'], label="显示窗口", info="实时调整参数", value=get_config_from_key('_is_show_top_window'))
                _aim_mod = gr.Radio(['1', '2', '3'], label="瞄准模式", info="1 左键 2右键 3左右", value=str(get_config_from_key('_aim_mod')))
            with gr.Row():  # 并行显示，可开多列
                with gr.Column():  # 并列显示，可开多行
                    with gr.Tab("权重相关参数"):
                        with gr.Row():  # 并行显示，可开多列
                            _conf_thres = gr.Slider(0, 1, step=0.001, value=get_config_from_key('_conf_thres'),
                                                    label=" 置信度",
                                                    info="置信度")  # 滑动条
                            _iou_thres = gr.Slider(0, 1, step=0.001, value=get_config_from_key('_iou_thres'),
                                                   label=" 交并集",
                                                   info="交并集")  # 滑动条
                            _weight = gr.Dropdown(models_files_list,
                                                  value=get_config_from_key('_weight'),
                                                  label="权重文件完整名称", info="选择已有的权重文件")
            with gr.Row():  # 并行显示，可开多列
                with gr.Column():  # 并列显示，可开多行
                    with gr.Tab("截图相关参数"):
                        with gr.Row():  # 并行显示，可开多列
                            _model_imgsz = gr.Radio(['320', '416', '640'], label="模型大小", info="训练时模型的大小",
                                                    value=str(get_config_from_key('_model_imgsz')))  # 单选
                            _grab_width = gr.Slider(1, 1920, value=get_config_from_key('_grab_width'), label="截图范围",
                                                    info="设置x轴截图范围值")  # 滑动条
                            _grab_height = gr.Slider(1, 1080, value=get_config_from_key('_grab_height'), label="截图范围",
                                                     info="设置y轴截图范围值")  # 滑动条
            with gr.Row():  # 并行显示，可开多列
                with gr.Column():  # 并列显示，可开多行
                    with gr.Column():  # 并列显示，可开多行
                        with gr.Tab("PID算法X轴设置"):
                            with gr.Row():  # 并行显示，可开多列
                                pid_x_P = gr.Slider(0, 1, step=0.001, value=get_config_from_key('pid_x_P'),
                                                    label=" P 值",
                                                    info="移动速度，值越高速度越快，也越抖")  # 滑动条
                                pid_x_I = gr.Slider(0, 1, step=0.001, value=get_config_from_key('pid_x_I'),
                                                    label=" I 值",
                                                    info="动态补偿，防止目标在移动时跟不上目标")  # 滑动条
                                pid_x_D = gr.Slider(0, 1, step=0.001, value=get_config_from_key('pid_x_D'),
                                                    label=" D 值",
                                                    info="抵消振荡，增加反应，振荡时适当增加")  # 滑动条
                            with gr.Row():  # 并行显示，可开多列
                                pid_x_min = gr.Slider(1, 10, step=1, value=get_config_from_key('pid_x_min'),
                                                      label="最小补偿阈值",
                                                      info="限制补偿，防止振荡")  # 滑动条
                                pid_x_max = gr.Slider(0, 250, step=1, value=get_config_from_key('pid_x_max'),
                                                      label="最大补偿阈值",
                                                      info="限制最大补偿阈值，避免补偿过头")  # 滑动条

                    with gr.Tab("PID算法Y轴设置"):
                        with gr.Row():  # 并行显示，可开多列
                            pid_y_P = gr.Slider(0, 1, step=0.001, value=get_config_from_key('pid_y_P'), label=" P 值",
                                                info="移动速度，值越高速度越快，也越抖")  # 滑动条
                            pid_y_I = gr.Slider(0, 1, step=0.001, value=get_config_from_key('pid_y_I'), label=" I 值",
                                                info="动态补偿，防止目标在移动时跟不上目标")  # 滑动条
                            pid_y_D = gr.Slider(0, 1, step=0.001, value=get_config_from_key('pid_y_D'), label=" D 值",
                                                info="抵消振荡，增加反应，振荡时适当增加")  # 滑动条
                        with gr.Row():  # 并行显示，可开多列
                            pid_y_min = gr.Slider(1, 10, step=1, value=get_config_from_key('pid_y_min'), label="最小补偿阈值",
                                                  info="限制补偿，防止振荡")  # 滑动条
                            pid_y_max = gr.Slider(0, 250, step=1, value=get_config_from_key('pid_y_max'),
                                                  label="最大补偿阈值",
                                                  info="限制最大补偿阈值，避免补偿过头")  # 滑动条
            bottom2 = gr.Button(value="保存")
            bottom2.click(submit,
                          inputs=[
                              _debug,
                              _is_show_top_window,
                              _conf_thres,
                              _iou_thres,
                              _weight,
                              _model_imgsz,
                              _grab_width,
                              _grab_height,
                              pid_x_P,
                              pid_x_I,
                              pid_x_D,
                              pid_x_min,
                              pid_x_max,
                              pid_y_P,
                              pid_y_I,
                              pid_y_D,
                              pid_y_min,
                              pid_y_max,
                          ])  # 触发
            bottom8 = gr.Button(value="重启软件")
            bottom8.click(restart_program)
    demo.launch()
