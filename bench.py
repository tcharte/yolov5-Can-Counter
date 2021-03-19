if __name__ == '__main__':
    import multiprocessing
    import queue
    import time
    import json

    from video_preloader import video_preloader
    from frame_processor import YoloTrainedModelFrameProcessor
    from calibration import multi_can_calibrate_frame_splits
    from count_cans_process import count_cans
    from cameras import LoopCamera
    from monitoring_process import monitor_performance

    # Video parameters:
    source = './data/mp4s/roughly_880_cans.mp4'

    # Calibration parameters
    lower_bound = 0.1
    upper_trigger = 0.7
    upper_bound = 0.9
    min_splits = 4

    # Network parameters
    conf_thresh = 0.25
    iou_thresh = 0.45

    # Benchmark parameters
    model_weights = {'yolov3-tiny': './weights/yolov3-tiny_best_of_600plus_epochs.pt', 'yolov5s': './weights/best_600_epoch_new_dataset.pt'}
    img_sizes = [320, 480, 640]
    camera_fps = [15, 30, 60]
    n_loops = 20

    models = []

def main():
    for fps in camera_fps:
        video = video_preloader(source, fps)
        complete_loop = len(video)
        for model, weights in model_weights.items():
            for img_size in img_sizes:
                network_params = (weights, img_size, conf_thresh, iou_thresh)
                frame_processor = YoloTrainedModelFrameProcessor(*network_params)
                splits = multi_can_calibrate_frame_splits(frame_processor, video, lower_bound, upper_trigger, upper_bound, min_splits)
                if len(splits) < min_splits:
                    return
                    
                print('Detection Model:', model, 'Image Size:', img_size, 'Camera FPS:', fps)
                frame_queue = multiprocessing.Queue(maxsize=100000)  # Queue for the instances to get frames from the class. Max ~800MB
                sorted_detections_queue = multiprocessing.Queue(maxsize=100000)
                count_queue = multiprocessing.Queue(maxsize=1)


                print('Initializing counting process...')
                count_process = multiprocessing.Process(target=count_cans, args=(splits, sorted_detections_queue, count_queue))
                count_process.start()
                
                print('Initializing camera...')
                loop_camera = LoopCamera(video, frame_queue, fps, process_trigger=True)  # Had to send a True signal so the process creation would not loop infinitely


                time.sleep(10) # Pausing so all processes have time to start
                tic = time.time()
                while True:
                    try:
                        frame, frame_id = frame_queue.get(block=False)
                        detections = frame_processor.detect_cans(frame)
                        sorted_detections_queue.put((detections, frame_id))

                        total_cans, current_frame_id = count_queue.get(block=False)
                        print('  frame', current_frame_id, '/', n_loops*complete_loop, '\t\tCounted Cans:', total_cans, end='\r')
                        if current_frame_id >= n_loops*complete_loop: # If n_loops are completed, calculate results and stop
                            toc = time.time()
                            total_time = toc-tic
                            calculated_fps = current_frame_id/total_time
                            if not loop_camera.real_count_queue.empty():
                                real = loop_camera.get_count()
                            else:
                                real = round((current_frame_id / complete_loop) * 880)
                            error = (total_cans - real) / real * 100
                            print('<<------------------COMPLETE------------------>>')
                            print('Model:', model, '-', img_size, '-', fps)
                            print('Time:', total_time)
                            print('Calculated fps:', calculated_fps)
                            print('Error', error)
                            print('<<-------------------------------------------->>')
                            json_data = {
                                'Model': model,
                                #'Weights File': weights,
                                'Image Size': img_size,
                                'Video FPS': fps,
                                'Calculated FPS': calculated_fps,
                                'Error': error
                            }
                            models.append(json_data)

                            del frame, frame_id, detections, total_cans, current_frame_id
                            break

                        del frame, frame_id, detections, total_cans, current_frame_id
                    except queue.Empty:
                        continue
                count_process.terminate()
                loop_camera.terminate()
                del network_params, frame_processor, splits, frame_queue, sorted_detections_queue, count_queue, loop_camera


        del video
    
    with open('benchmark.json', 'w') as outfile:
        json.dump(models, outfile)

    

if __name__ == '__main__':
    main()