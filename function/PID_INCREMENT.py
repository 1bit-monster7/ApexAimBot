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
