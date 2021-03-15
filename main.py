if __name__ == '__main__':
    import multiprocessing
    import queue

    from video_preloader import video_preloader
    from cameras import LoopCamera
    from calibration import multi_can_calibrate_frame_splits
    from frame_processor import YoloTrainedModelFrameProcessor
    from detections_sorter_process import sort_detections
    from count_cans_process import count_cans
    from monitoring_process import monitor_performance

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
    num_models = 1
    # weights = './weights/best_600_epoch_new_dataset.pt'
    weights = './weights/yolov3-tiny_best_of_600plus_epochs.pt'
    network_params = (weights, img_size, conf_thresh, iou_thresh)


def main():
    frame_processor = YoloTrainedModelFrameProcessor(*network_params)
    video = video_preloader(calibration_source, fps)
    complete_loop = len(video)
    splits = multi_can_calibrate_frame_splits(frame_processor, video, lower_bound, upper_trigger, upper_bound, min_splits)
    print('Splits:', splits)
    if len(splits) < min_splits:
        return

    frame_queue = multiprocessing.Queue(maxsize=100000)  # Queue for the instances to get frames from the class. Max ~800MB
    raw_detections_queue = multiprocessing.Queue(maxsize=100000)
    sorted_detections_queue = multiprocessing.Queue(maxsize=100000)
    count_queue = multiprocessing.Queue(maxsize=1)

    print('Initializing counting process...')
    count_process = multiprocessing.Process(target=count_cans, args=(splits, sorted_detections_queue, count_queue))
    count_process.start()

    print('Initializing detections sorting process...')
    detections_sorter_process = multiprocessing.Process(target=sort_detections, args=(raw_detections_queue, sorted_detections_queue, splits, num_models))
    detections_sorter_process.start()

    print('Initializing camera...')
    loop_camera = LoopCamera(video, frame_queue, fps, process_trigger=True)  # Had to send a True signal so the process creation would not loop infinitely

    print('Initializing Monitoring Process...')
    monitoring_process = multiprocessing.Process(target=monitor_performance, args=(loop_camera, count_queue, frame_queue, raw_detections_queue, sorted_detections_queue, fps, complete_loop))
    monitoring_process.start()

    print('Detection process initialized!')
    while True:
        try:
            frame, frame_id = frame_queue.get(block=False)
            detections = frame_processor.detect_cans(frame)
            raw_detections_queue.put((detections, frame_id))
            del frame
            del frame_id
            del detections
        except queue.Empty:
            continue


if __name__ == '__main__':
    main()
