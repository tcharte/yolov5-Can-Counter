def detect_cans(frame_processor, frame_queue, detections_queue, params):
    from queue import Empty
    import platform
    if platform.system() == 'Linux':
        print('Detection process initialized!')
        while True:
            try:
                frame, frame_id = frame_queue.get(block=False)
                detections = frame_processor.detect_cans(frame)
                detections_queue.put((detections, frame_id))
                del frame
                del frame_id
                del detections
            except Empty:
                continue

    elif platform.system() == 'Windows':
        from frame_processor import YoloTrainedModelFrameProcessor
        frame_processor = YoloTrainedModelFrameProcessor(*params)
        print('Detection process initialized!')
        while True:
            try:
                frame, frame_id = frame_queue.get(block=False)
                detections = frame_processor.detect_cans(frame)
                detections_queue.put((detections, frame_id))
                del frame
                del frame_id
                del detections
            except Empty:
                continue



