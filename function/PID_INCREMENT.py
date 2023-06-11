# 增量式PID系统
class PID_PLUS(object):
    def __init__(self, exp_val, P: float, I: float, D: float):
        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.PIDOutput = 0.0  # PID控制器输出
        self.SystemOutput = 0.0  # 系统输出值
        self.LastSystemOutput = 0.0  # 系统的上一次输出

        self.Error = 0.0
        self.LastError = 0.0
        self.LastLastError = 0.0

    # 设置PID控制器参数
    def getMove(self, sub_s):
        self.Error = sub_s - self.SystemOutput
        # 计算增量
        IncrementalValue = self.Kp * (self.Error - self.LastError) \
                           + self.Ki * self.Error + self.Kd * (self.Error - 2 * self.LastError + self.LastLastError)
        # 计算输出
        self.PIDOutput += IncrementalValue
        self.LastLastError = self.LastError
        self.LastError = self.Error
        return self.PIDOutput


class PID_PLUS_PLUS(object):
    def __init__(self, exp_val, P: float, I: float, D: float):
        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.PIDOutput = 0.0  # PID控制器输出
        self.SystemOutput = 0.0  # 系统输出值
        self.LastSystemOutput = 0.0  # 系统的上一次输出

        self.Error = 0.0
        self.LastError = 0.0
        self.LastLastError = 0.0

    # 设置PID控制器参数
    def getMove(self, sub_s, max_output=None):
        self.Error = sub_s - self.SystemOutput
        # 计算增量
        IncrementalValue = self.Kp * (self.Error - self.LastError) \
                           + self.Ki * self.Error + self.Kd * (self.Error - 2 * self.LastError + self.LastLastError)
        # 计算输出
        self.PIDOutput += IncrementalValue

        # 限制输出值在最大步长范围内 (如果 max_output 参数提供)
        if max_output is not None:
            self.PIDOutput = max(min(self.PIDOutput, max_output), -max_output)

        self.LastLastError = self.LastError
        self.LastError = self.Error
        return self.PIDOutput


class ADRC_PLUS(object):
    def __init__(self, exp_val, h: float, r: float, ki: float, kp: float, kd: float, b0: float):
        self.exp_val = exp_val  # 期望值
        self.h = h  # 观测器增益参数
        self.r = r  # 扰动观测器增益参数
        self.ki = ki  # 控制器积分增益参数
        self.kp = kp  # 控制器比例增益参数
        self.kd = kd  # 控制器微分增益参数

        self.b0 = b0  # 扰动估计器增益参数，初始值设置为系统直流增益

        self.PIDOutput = 0.0  # PID控制器输出
        self.SystemOutput = 0.0  # 系统输出值
        self.LastSystemOutput = 0.0  # 系统的上一次输出

        self.Error = 0.0  # 当前误差
        self.LastError = 0.0  # 上一个时刻的误差
        self.IntegralError = 0.0  # 误差积分值

        self.DerivativeError = 0.0  # 误差微分值
        self.LastDerivativeError = 0.0  # 上一个时刻的微分值

        self.DisturbanceEstimate = 0.0  # 扰动估计器估计值
        self.LastDisturbanceEstimate = 0.0  # 上一个时刻的扰动估计器估计值

        self.SystemOutputEstimate = 0.0 # 系统输出值观测值
        self.LastSystemOutputEstimate = 0.0 # 上一个时刻的系统输出值观测值

    # 设置ADRC控制器参数
    def getMove(self, sub_s, max_output=None):

        # 第1步，通过扰动估计器估计系统中的扰动信号
        DisturbanceEstimate = self.LastDisturbanceEstimate + self.b0 * (self.Error - self.LastError)

        # 第2步，通过观测器估计系统输出值和其一阶导数
        SystemOutputEstimate = self.SystemOutput + self.h * (self.PIDOutput + DisturbanceEstimate - self.ki * self.IntegralError)
        DerivativeError = self.r * (SystemOutputEstimate - self.LastSystemOutputEstimate) + (1 - self.r) * self.LastDerivativeError

        # 第3步，计算PID控制器输出
        self.PIDOutput = self.kp * (self.exp_val - SystemOutputEstimate) + self.ki * self.IntegralError \
                         + self.kd * DerivativeError

        # 第4步，限制输出值在最大步长范围内 (如果 max_output 参数提供)
        if max_output is not None:
            self.PIDOutput = max(min(self.PIDOutput, max_output), -max_output)

        # 更新状态变量
        self.LastError = self.Error
        self.Error = sub_s - self.SystemOutput
        self.IntegralError += self.Error
        self.LastDerivativeError = DerivativeError
        self.LastDisturbanceEstimate = DisturbanceEstimate
        self.LastSystemOutputEstimate = SystemOutputEstimate

        return self.PIDOutput