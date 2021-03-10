if __name__ == '__main__':
    import multiprocessing
    import time
    import os
    import queue

    from video_preloader import video_preloader
    from cameras import LoopCamera
    from calibration import multi_can_calibrate_frame_splits
    from frame_processor import YoloV5sTrainedModelFrameProcessor
    from detections_sorter_process import sort_detections
    from count_cans_process import count_cans
    from detect_process import detect_cans

    print('Starting CanSoft...')
    # Video parameters:
    calibration_source = './data/mp4s/roughly_880_cans.mp4'
    test_source = './data/mp4s/roughly_880_cans.mp4'
    fps = 60

    # Calibration parameters
    lower_bound = 0.1
    upper_trigger = 0.7
    upper_bound = 0.9
    min_splits = 4

    # Network parameters
    img_size = 640
    conf_thresh = 0.25
    iou_thresh = 0.45
    num_models = 2
    weights = './weights/best_600_epoch_new_dataset.pt'
    network_params = (weights, img_size, conf_thresh, iou_thresh)


def main():
    frame_processor = YoloV5sTrainedModelFrameProcessor(*network_params)
    video = video_preloader(calibration_source, fps)
    splits = multi_can_calibrate_frame_splits(frame_processor, video, lower_bound, upper_trigger, upper_bound, min_splits)
    print('Splits:', splits)
    if len(splits) < min_splits:
        return

    frame_queue = multiprocessing.Queue(maxsize=1000)  # Queue for the instances to get frames from the class. Max ~800MB
    raw_detections_queue = multiprocessing.Queue(maxsize=100000)
    sorted_detections_queue = multiprocessing.Queue(maxsize=100000)
    count_queue = multiprocessing.Queue(maxsize=1)

    print('Initializing counting process...')
    count_process = multiprocessing.Process(target=count_cans, args=(splits, sorted_detections_queue, count_queue))
    count_process.start()

    print('Initializing detections sorting process...')
    detections_sorter_process = multiprocessing.Process(target=sort_detections, args=(raw_detections_queue, sorted_detections_queue, splits, num_models))
    detections_sorter_process.start()

    print('Initializing', num_models, 'detection processes...')
    for _ in range(num_models):
        detection_process = multiprocessing.Process(target=detect_cans, args=(frame_queue, raw_detections_queue, network_params))
        detection_process.start()

    time.sleep(10)
    print('Initializing camera...')
    loop_camera = LoopCamera(video, frame_queue, fps, process_trigger=True)  # Had to send a True signal so the process creation would not loop infinitely

    print('Ready to total up the counted cans...')
    refresh_time = 1
    actual_can_count = 'N/A'
    error = 'N/A'
    start = time.time_ns()/1e9
    t1 = time.time_ns()/1e9
    start_id = 0
    while True:
        try:
            total_cans, current_frame_id = count_queue.get(block=False)
            t2 = time.time_ns() / 1e9
            elapsed_time = t2 - t1
            total_time = t2 - start
            if elapsed_time >= refresh_time:
                clear = lambda: os.system('cls')
                clear()
                print('-Performance-')
                print("Execution time: %02d:%02d:%02d:%02d" % (total_time // 86400, total_time // 3600 % 24, total_time // 60 % 60, total_time % 60))
                print('Current total cans:', total_cans)
                if not loop_camera.real_count_queue.empty():
                    actual = loop_camera.get_count()
                    error = (total_cans - actual) / actual * 100
                    actual_can_count = actual
                print('Actual can count:', actual_can_count)
                print('Error (%):', error)
                print('Measured FPS:', (current_frame_id - start_id) / elapsed_time)
                print('Camera FPS:', fps)
                print('Parallel detection processes:', num_models)
                print('Backed-up frames:', frame_queue.qsize())
                print('Backed-up unsorted detections:', raw_detections_queue.qsize())
                print('Backed-up sorted detections:', sorted_detections_queue.qsize())
                print('Backed-up counts:', count_queue.qsize())
                t1 = time.time_ns() / 1e9
                start_id = current_frame_id
        except queue.Empty:
            continue


if __name__ == '__main__':
    main()
