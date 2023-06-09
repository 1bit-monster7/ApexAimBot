from ctypes import windll

timeBeginPeriod = windll.winmm.timeBeginPeriod
timeEndPeriod = windll.winmm.timeEndPeriod
Sleep = windll.kernel32.Sleep


def delay_ms(ms):
    timeBeginPeriod(1)
    Sleep(int(ms))
    timeEndPeriod(1)
