def detect_cans(frame_queue, detections_queue, params):
    from frame_processor import TrainedModelFrameProcessor
    frame_processor = TrainedModelFrameProcessor(*params)
    print('Detection process initialized!')
    while True:
        frame, frame_id = frame_queue.get()
        detections = frame_processor.detect_cans(frame)
        detections_queue.put((detections, frame_id))

