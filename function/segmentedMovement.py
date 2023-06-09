import random

from function.FOV import FOV_X, FOV_Y
from function.logitech import Logitech

last_x, last_y = 0, 0


def _mouse(x, y):
    Logitech.mouse.move(x, y)


def segmented_movement_x(x, _min=10, _max=10, offset_y=0):
    if abs(x) == 0 and abs(offset_y) == 0:
        return

    _mouse(0, offset_y)

    _to_x = int(x >= 0)
    _random = get_random_integer(_min, _max)

    _total_x, _remainder_x = cutting_num(x, _random)

    total_move = _total_x

    for i in range(total_move):
        if i < _total_x:
            _real_x = get_num_from_random(_to_x, _random)
        else:
            _real_x = 0
        _mouse(int(FOV_X(_real_x)), 0)

    if abs(_remainder_x) > 0:
        _mouse(int(FOV_X(_remainder_x)), 0)


def segmented_movement_xy(x, y, _min=10, _max=10):
    if abs(x) == 0 and abs(y) == 0:
        return

    _to_x = int(x >= 0)
    _to_y = int(y >= 0)
    _random = get_random_integer(_min, _max)

    _total_x, _remainder_x = cutting_num(x, _random)
    sign_y = -1 if y < 0 else 1
    _total_y, _remainder_y = cutting_num(y, _random)

    total_move = max(_total_x, _total_y)

    for i in range(total_move):
        if i < _total_x:
            _real_x = get_num_from_random(_to_x, _random)
        else:
            _real_x = 0

        if i < _total_y:
            if i == _total_y - 1:
                _real_y = sign_y * get_num_from_random(1, abs(_remainder_y))
            else:
                _real_y = sign_y * _random
        else:
            _real_y = 0

        _mouse(int(FOV_X(_real_x)), int(FOV_Y(_real_y)))

    if abs(_remainder_x) > 0:
        _mouse(int(FOV_X(_remainder_x)), 0)
    if abs(_remainder_y) > 0:
        _mouse(0, int(FOV_Y(_remainder_y)))


def get_num_from_random(_to, random_num):
    if _to > 0:
        return random_num
    else:
        return -random_num


def get_random_integer(lower, upper):
    """
    生成指定范围内的随机整数，包含端点值
    :param lower: 范围下限值
    :param upper: 范围上限值
    :return: 随机整数
    """
    return random.randint(lower, upper)


def cutting_num(num, cut):
    """
    计算一个整数可以最多切割多少次，并返回剩余数
    :param num: 待切割的整数
    :param cut: 每次切割的数
    :return: 最多切割次数和剩余数的元组
    """
    if num >= 0:
        # 如果 num 为非负数，则计算最多切割次数和剩余数
        times = num // cut  # 整除，得到最多可以切割的次数
        residue = num % cut  # 取余，得到剩余数
    else:
        # 如果 num 为负数，则先将它变为正数进行计算
        positive_num = -num
        times = positive_num // cut
        residue = -(positive_num % cut)  # 剩余数也要变成负数

    return times, residue
