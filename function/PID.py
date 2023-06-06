class PID(object):
    def __init__(self, kp, ki, kd, smoothness):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.smoothness = smoothness
        self.p = 0
        self.i = 0
        self.d = 0
        self.last = 0
        self.p_last = 0
        self.i_last = 0
        self.d_last = 0

    def cmd_pid(self, err, dt):
        self.p = err * self.Kp
        self.i = self.i + err * self.Ki * dt
        self.d = (err - self.last) * self.Kd / dt
        self.last = err

        p = self.p * self.smoothness + (1 - self.smoothness) * self.p_last
        i = self.i * self.smoothness + (1 - self.smoothness) * self.i_last
        d = self.d * self.smoothness + (1 - self.smoothness) * self.d_last
        self.p_last = p
        self.i_last = i
        self.d_last = d

        output = p + i + d
        return output

    def clear_i_term(self):
        self.i = 0
        self.i_last = 0
