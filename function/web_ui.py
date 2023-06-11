import configparser
import os
import sys
import time

import gradio as gr

from G import params_list, dark_themes
from function.configUtils import get_ini, set_config, _write

config = configparser.ConfigParser()
# 从1bit.ai.config文件中读取参数和值
config.read("1bit.ai.config")

textGroup = 'group'  # 分组名称

notice_god = None


# 定义一个公共字典对象 用于保存参数 这里的数组key顺序要和下面的传参顺序一致


def submit(*args):
    print(f"保存并拿到参数keys数组：{args}")
    global notice_god
    count = 0
    for val in args:
        key = params_list[count]  # key
        count += 1
        set_config(key, val)
    _write()  # 写入ini
    time.sleep(0.5)
    notice_god.notify(f"注意：已修改参数 将根据最新的参数进行程序执行")


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


def create_ui(BIT_GOD):
    global notice_god
    notice_god = BIT_GOD
    models_files_list = list_weights_files()  # 获取所有权重文件
    # 组件
    shake_coefficient = gr.Slider(0, 100, step=0.1, value=get_ini('shake_coefficient'), label=" 抖枪系数", info="抖枪系数 一般0.1 微调")  # 滑动条
    shake_coefficient_y = gr.Slider(0, 100, step=0.1, value=get_ini('shake_coefficient_y'), label=" 抖枪系数Y轴", info="抖枪系数 一般0.1 微调")  # 滑动条
    shake_delay = gr.Slider(0, 100, step=0.1, value=get_ini('shake_delay'), label="抖动延时", info="抖枪延时 值越小抖动速度越快")  # 滑动条
    grab_window_title = gr.inputs.Textbox(label="游戏名称", default=get_ini('grab_window_title'))
    screen_width = gr.inputs.Textbox(label="屏幕分辨率X", default=get_ini('screen_width'))
    screen_height = gr.inputs.Textbox(label="屏幕分辨率Y", default=get_ini('screen_height'))
    debug = gr.Radio(['1', '0'], label="Debug", info="保存时是否重新载入模型 1 是开启 0 是关闭", value=str(get_ini('debug')))
    is_show_top_window = gr.Radio(['1', '0'], label="显示窗口", info="右上角win窗口 1 是开启 0 是关闭", value=str(get_ini('is_show_top_window')))
    aim_mod = gr.Radio(['0', '1', '2'], label="瞄准模式", info="1 左键 2右键 3左右", value=str(get_ini('aim_mod')))
    model_imgsz = gr.Radio(['320', '416', '640'], label="模型大小", info="训练时模型的大小", value=str(get_ini('model_imgsz')))  # 单选
    grab_width = gr.Slider(0, 1920, value=get_ini('grab_width'), label="截图范围", info="设置x轴截图范围值")  # 滑动条
    grab_height = gr.Slider(0, 1080, value=get_ini('grab_height'), label="截图范围", info="设置y轴截图范围值")  # 滑动条
    min_step = gr.Slider(0, 100, step=1, value=get_ini('min_step'), label="腰射最大移动像素值", info="腰射的最大可移动像素值")  # 滑动条
    max_step = gr.Slider(0, 200, step=1, value=get_ini('max_step'), label="开镜最大移动像素值", info="右键瞄准时的最大可移动像素值")  # 滑动条
    sens = gr.Slider(0, 10, step=0.001, value=get_ini('sens'), label=" 鼠标速度", info="Apex 游戏设置里面鼠标速度")  # 滑动条
    ads = gr.Slider(0, 10, step=0.001, value=get_ini('ads'), label=" 开镜鼠标速度", info="ads Apex游戏内 开镜速度")  # 滑动条
    modifier_value = gr.Slider(0, 1, step=0.001, value=str(get_ini('modifier_value')), label="最终计算出的压枪系数 一般不需要手动调整", info="0 - 1 越小越压")  # 滑动条
    conf_thres = gr.Slider(0, 1, step=0.001, value=get_ini('conf_thres'), label=" 置信度", info="置信度")  # 滑动条
    weight = gr.Dropdown(models_files_list, value=get_ini('weight'), label="权重文件", info="选择权重文件")
    iou_thres = gr.Slider(0, 1, step=0.001, value=get_ini('iou_thres'), label=" 交并集", info="交并集")  # 滑动条
    pid_x_p = gr.Slider(0, 1, step=0.001, value=get_ini('pid_x_p'), label=" P参数（比例系数）", info="设置X轴的P参数")  # 滑动条
    pid_x_i = gr.Slider(0, 1, step=0.001, value=get_ini('pid_x_i'), label=" I 参数（积分系数）", info="设置X轴的I参数")  # 滑动条
    pid_x_d = gr.Slider(0, 1, step=0.001, value=get_ini('pid_x_d'), label=" D 参数（微分系数）", info="设置X轴的D参数")  # 滑动条
    pid_y_p = gr.Slider(0, 1, step=0.001, value=get_ini('pid_y_p'), label=" P 参数（比例系数）", info="设置Y轴的P参数")  # 滑动条
    pid_y_i = gr.Slider(0, 1, step=0.001, value=get_ini('pid_y_i'), label=" I 参数（积分系数）", info="设置Y轴的I参数")  # 滑动条
    pid_y_d = gr.Slider(0, 1, step=0.001, value=get_ini('pid_y_d'), label=" D 参数（微分系数）", info="设置Y轴的D参数")  # 滑动条

    # render
    with gr.Blocks(
            css=".gradio-container {background-color: #03001C;max-width:100vw!important;margin:0!important;}",
            theme=gr.themes.Soft(primary_hue=dark_themes)) as demo:
        with gr.Tab("From 1bit q1748244285"):
            with gr.Row():
                with gr.Tab("基本设置"):
                    with gr.Row():
                        debug.render()
                        is_show_top_window.render()
                        aim_mod.render()
            with gr.Row():
                with gr.Tab("抖枪设置"):
                    with gr.Row():
                        shake_coefficient.render()
                        shake_coefficient_y.render()
                        shake_delay.render()
            with gr.Row():
                with gr.Tab("自识别压枪设置"):
                    with gr.Row():
                        sens.render()
                        ads.render()
                        modifier_value.render()
            with gr.Row():
                with gr.Tab("屏幕设置"):
                    with gr.Row():
                        grab_window_title.render()
                        screen_width.render()
                        screen_height.render()
            with gr.Row():
                with gr.Tab("截图设置"):
                    with gr.Row():
                        model_imgsz.render()
                        grab_width.render()
                        grab_height.render()
            with gr.Row():
                with gr.Tab("移动设置"):
                    with gr.Row():
                        min_step.render()
                        max_step.render()
            with gr.Row():
                with gr.Tab("模型设置"):
                    with gr.Row():
                        conf_thres.render()
                        weight.render()
                        iou_thres.render()
            with gr.Row():
                with gr.Tab("PID算法X轴设置"):
                    with gr.Row():
                        pid_x_p.render()
                        pid_x_i.render()
                        pid_x_d.render()
            with gr.Row():
                with gr.Tab("PID算法Y轴设置"):
                    with gr.Row():
                        pid_y_p.render()
                        pid_y_i.render()
                        pid_y_d.render()
        bottom2 = gr.Button(value="保存")
        bottom2.click(submit, inputs=[
            shake_coefficient,
            shake_coefficient_y,
            shake_delay,
            sens,
            ads,
            min_step,
            max_step,
            grab_window_title,
            screen_width,
            screen_height,
            debug,
            is_show_top_window,
            aim_mod,
            conf_thres,
            iou_thres,
            weight,
            model_imgsz,
            grab_width,
            grab_height,
            pid_x_p,
            pid_x_i,
            pid_x_d,
            pid_y_p,
            pid_y_i,
            pid_y_d,
            modifier_value
        ])  # 触发
        bottom8 = gr.Button(value="重启软件")
        bottom8.click(restart_program)
    demo.launch(inbrowser=True)
