def frame_generator(video, frame_queue, count_queue, fps):
    import time
    """
    Target function for running the camera acquisition as a process. This makes more sense as the dedicated process for
    the camera ensures it is not slowed down by the rest of the program
    """
    print('Pausing camera so all processes have time to start')
    time.sleep(10)
    print('Loop camera started!')
    total_count = 0
    frame_id = 0
    while True:
        for frame in video:
            if frame_queue.full():
                print('Frame queue is full! Consumers are too slow. Can count is no longer accurate! Skipping frame...')
            else:
                frame_queue.put((frame, frame_id))
                frame_id += 1
                time.sleep(1/fps)

        total_count += 880

        if count_queue.full():
            print('Count queue is full!def frame_generator(video, frame_queue, count_queue, fps):
    import time
    """
    Target function for running the camera acquisition as a process. This makes more sense as the dedicated process for
    the camera ensures it is not slowed down by the rest of the program
    """
    print('Pausing camera so all processes have time to start')
    time.sleep(10)
    print('Loop camera started!')
    total_count = 0
    frame_id = 0
    while True:
        for frame in video:
            if frame_queue.full():
                print('Frame queue is full! Consumers are too slow. Can count is no longer accurate! Skipping frame...')
            else:
                frame_queue.put((frame, frame_id))
                frame_id += 1
                time.sleep(1/fps)

        total_count += 880

        if count_queue.full():
            print('Count queue is full! Consumers are too slow. The actual can count is no longer accurate...')
        else:
            count_queue.put(total_count)


class LoopCamera:
    import multiprocessing
    """
    Simple looping camera that uses the footage of a known amount of cans. The camera will track the can count internally
    to measure against the calculated can count
    """
    real_count_queue = multiprocessing.Queue(maxsize=100000)  # Max will be quite small since this holds integers
    generator = None

    def __init__(self, video, frame_queue, fps, process_trigger=False):
        if process_trigger:  # Introduced a flag that only the main process can trigger so infinite process do not spawn
            if self.generator is None:
                import multiprocessing
                self.generator = multiprocessing.Process(target=frame_generator, args=(video, frame_queue, self.real_count_queue, fps))
                self.generator.start()

    def get_count(self):
        return self.real_count_queue.get()

    def terminate(self):
        self.generator.terminate()
        return
 Consumers are too slow. The actual can count is no longer accurate...')
        else:
            count_queue.put(total_count)


class LoopCamera:
    import multiprocessing
    """
    Simple looping camera that uses the footage of a known amount of cans. The camera will track the can count internally
    to measure against the calculated can count
    """
    real_count_queue = multiprocessing.Queue(maxsize=100000)  # Max will be quite small since this holds integers
    generator = None

    def __init__(self, video, frame_queue, fps, process_trigger=False):
        if process_trigger:  # Introduced a flag that only the main process can trigger so infinite process do not spawn
            if self.generator is None:
                import multiprocessing
                self.generator = multiprocessing.Process(target=frame_generator, args=(video, frame_queue, self.real_count_queue, fps))
                self.generator.start()

    def get_count(self):
        return self.real_count_queue.get()

    def terminate(self):
        self.generator.terminate()
        return
