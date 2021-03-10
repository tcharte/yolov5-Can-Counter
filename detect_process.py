def detect_cans(frame_queue, detections_queue, params):
    from frame_processor import YoloV5sTrainedModelFrameProcessor
    from queue import Empty
    frame_processor = YoloV5sTrainedModelFrameProcessor(*params)
    print('Detection process initialized!')
    while True:
        try:
            frame, frame_id = frame_queue.get(block=False)
            detections = frame_processor.detect_cans(frame)
            detections_queue.put((detections, frame_id))
        except Empty:
            continue

