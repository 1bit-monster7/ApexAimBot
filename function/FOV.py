# DPI
import math

pixels_360 = 3273  # 一圈总像素

# pixels_per_degree = pixels_360 / 360


screen_width = 1920
cl_fovScale = 1.46357
screen_height = 1080
desired_fov = 104  # 视野
pi = 3.141592653589793


def FOV_X(delta_x):
    fov = 117.863927
    # FOV传入游戏的视野值
    width = 1920
    pixel_x = 3272.72
    # 改进per_pixel_rad的计算
    per_pixel_rad = pixel_x / (2 * pi)

    delta_abs_x = abs(delta_x)

    sup_distance = (width / 2) / math.tan((fov * pi / 180) / 2)  # 角距离公式

    target_angle_rad = math.atan(delta_abs_x / sup_distance)

    target_move = target_angle_rad * per_pixel_rad

    if delta_x < 0:
        return -target_move
    else:
        return target_move


def FOV_Y(delta_y):
    fov = 86.069163

    height = 1080

    pixel_y = 1636.3636

    # 改进per_pixel_rad的计算
    per_pixel_rad = pixel_y / (2 * pi)

    delta_abs_y = abs(delta_y)

    sup_distance = (height / 2) / math.tan((fov * pi / 180) / 2)

    target_angle_rad = math.atan(delta_abs_y / sup_distance)

    target_move = target_angle_rad * per_pixel_rad

    if delta_y < 0:
        return -target_move
    else:
        return target_move
