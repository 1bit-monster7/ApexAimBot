import time

import dxshot


shape = 640
left, top = (1920 - shape) // 2, (1080 - shape) // 2
right, bottom = left + shape, top + shape
region = (left, top, right, bottom)
title = "[DXcam] FPS benchmark"
cam = dxshot.create()
start_time = time.perf_counter()
fps = 0
total_time = 0
while fps < 1000:
    start = time.perf_counter()
    frame = cam.grab(region=region)
    # frame = grab_gpt(window_title=grab_window_title, grab_rect=region)
    if frame is not None:
        now_time = time.perf_counter()
        print(f"{(now_time - start) * 1000}ms")
        fps += 1
        total_time += (now_time - start)
        # cv2.imshow('', frame)
        # cv2.waitKey(1)

end_time = time.perf_counter() - start_time

print(f"{title}: {fps / total_time}")
del cam