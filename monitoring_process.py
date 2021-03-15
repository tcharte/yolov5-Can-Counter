def monitor_performance(loop_camera, count_queue, frame_queue, sorted_detections_queue, fps, complete_loop):
    import time
    import os
    import platform
    import queue

    refresh_time = 1
    start = time.time()
    t1 = time.time()
    start_id = 0
    while True:
        try:
            total_cans, current_frame_id = count_queue.get(block=False)
            t2 = time.time()
            elapsed_time = t2 - t1
            total_time = t2 - start
            if elapsed_time >= refresh_time:
                if platform.system() == 'Windows':
                    clear = lambda: os.system('cls')
                    clear()
                elif platform.system() == 'Linux':
                    clear = lambda: os.system('clear')
                    clear()
                print('-Performance-')
                print("Execution time: %02d:%02d:%02d:%02d" % (
                total_time // 86400, total_time // 3600 % 24, total_time // 60 % 60, total_time % 60))
                print('Current total cans:', total_cans)
                if not loop_camera.real_count_queue.empty():
                    actual = loop_camera.get_count()
                    error = (total_cans - actual) / actual * 100
                    actual_can_count = actual
                else:
                    actual_can_count = round((current_frame_id / complete_loop) * 880)
                    error = (total_cans - actual_can_count) / actual_can_count * 100
                print('Actual can count:', actual_can_count)
                print('Error (%):', round(error, 2))
                print('Measured FPS:', round((current_frame_id - start_id) / elapsed_time))
                print('Camera FPS:', fps)
                print('Backed-up frames:', frame_queue.qsize())
                print('Backed-up unsorted detections:', raw_detections_queue.qsize())
                print('Backed-up sorted detections:', sorted_detections_queue.qsize())
                print('Backed-up counts:', count_queue.qsize())
                t1 = time.time()
                start_id = current_frame_id
        except queue.Empty:
            continue
