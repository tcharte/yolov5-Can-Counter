if __name__ == '__main__':
    import multiprocessing
    import time
    import os
    import platform
    import queue

    import torch.multiprocessing as mp

    if platform.system() == 'Linux':
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
    if platform.system() == 'Windows':
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass

    from video_preloader import video_preloader
    from cameras import LoopCamera
    from calibration import multi_can_calibrate_frame_splits
    from frame_processor import YoloTrainedModelFrameProcessor
    from detections_sorter_process import sort_detections
    from count_cans_process import count_cans
    from detect_process import detect_cans

    print('Starting CanSoft...')
    # Video parameters:
    # calibration_source = './data/mp4s/90fps.mp4'
    # test_source = './data/mp4s/90fps.mp4'
    calibration_source = './data/mp4s/1078.mp4'
    test_source = './data/mp4s/1078.mp4'
    fps = 60
    loop = False

    # Calibration parameters
    lower_bound = 0.2
    upper_trigger = 0.65
    upper_bound = 0.75
    min_splits = 5

    # Network parameters
    img_size = 640
    conf_thresh = 0.5
    iou_thresh = 0.45
    num_models = 2
    # weights = './weights/best_600_epoch_new_dataset.pt'
    weights = './weights./best_yolov3_tiny_600ep.pt'
    # weights = './weights/yolov3-tiny_best_of_600plus_epochs.pt'
    network_params = (weights, img_size, conf_thresh, iou_thresh)


def main():
    frame_processor = YoloTrainedModelFrameProcessor(*network_params)
    video = video_preloader(calibration_source, fps)
    complete_loop = len(video)
    splits = multi_can_calibrate_frame_splits(frame_processor, video, lower_bound, upper_trigger, upper_bound,
                                              min_splits)
    print('Splits:', splits)
    if len(splits) < min_splits:
        return

    frame_queue = None
    raw_detections_queue = None
    sorted_detections_queue = None
    count_queue = None

    if platform.system() == 'Linux':
        frame_queue = mp.Queue(maxsize=100000)  # Queue for the instances to get frames from the class. Max ~800MB
        raw_detections_queue = mp.Queue(maxsize=100000)
        sorted_detections_queue = mp.Queue(maxsize=100000)
        count_queue = mp.Queue(maxsize=1)
    elif platform.system() == 'Windows':
        frame_queue = multiprocessing.Queue(
            maxsize=100000)  # Queue for the instances to get frames from the class. Max ~800MB
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
    if platform.system() == 'Linux':
        for _ in range(num_models):
            detection_process = mp.Process(target=detect_cans,
                                           args=(frame_processor, frame_queue, raw_detections_queue, network_params))
            detection_process.start()

    elif platform.system() == 'Windows':
        for _ in range(num_models):
            detection_process = multiprocessing.Process(target=detect_cans,
                                                        args=(None, frame_queue, raw_detections_queue, network_params))
            detection_process.start()
    time.sleep(10)
    print('Initializing camera...')
    loop_camera = LoopCamera(video, frame_queue, fps, process_trigger=True,
                             loop=loop)  # Had to send a True signal so the process creation would not loop infinitely

    print('Ready to total up the counted cans...')
    refresh_time = 1
    start = time.time()
    t1 = time.time()
    total_cans = 0
    while True:
        try:
            total_cans, _ = count_queue.get(block=False)

        except queue.Empty:
            pass
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
            print("Execution time: %02d:%02d:%02d:%02d" % (total_time // 86400, total_time // 3600 % 24, total_time // 60 % 60, total_time % 60))
            print('Current total cans:', total_cans)
            print('Camera FPS:', fps)
            print('Parallel detection processes:', num_models)
            print('Backed-up frames:', frame_queue.qsize())
            print('Backed-up unsorted detections:', raw_detections_queue.qsize())
            print('Backed-up sorted detections:', sorted_detections_queue.qsize())
            print('Backed-up counts:', count_queue.qsize())
            t1 = time.time()


if __name__ == '__main__':
    main()


# detect.py --source ./data/mp4s/555.mp4 --weights ./weights/best_600_epoch_new_dataset.pt --view-img --conf-thres 0.9
