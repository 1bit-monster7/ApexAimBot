import random

from function.logitech import Logitech

last_x, last_y = 0, 0


def _mouse(x, y):
    Logitech.mouse.move(x, y)


def divide_with_floor(numerator, denominator):
    if numerator < 0:  # 如果被除数为负数
        numerator = abs(numerator)

    quotient = numerator // denominator
    remainder = abs(numerator) % denominator

    if numerator < 0:  # 如果被除数为负数
        remainder = denominator - remainder

    return quotient, remainder


def generate_random_int(lower_bound, upper_bound):
    """
    生成随机整数
    :return: 随机整数
    """
    return random.randint(lower_bound, upper_bound)


def segmented_movement_x(x, y, step):
    total, remainder = divide_with_floor(x, step)
    # print(f"x:{x} 每次移动的像素值:{step} 移动总次数:{total} 余数:{remainder} before")
    step = -step if x < 0 else step
    remainder = -remainder if x < 0 else remainder
    # print(f"x:{x} 每次移动的像素值:{step} 移动总次数:{total} 余数:{remainder}")
    total_step = 0
    for i in range(total):
        _mouse(step, 0)
        total_step += 1
        print(f"已移动 {abs(total_step)} 步。 每次移动{step}像素")
        # delay_ms(0)
    _mouse(remainder, y)
    # print(f'移动一次余数{remainder}和y轴')


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
