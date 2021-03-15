import queue
import time
from multiprocessing import Pool, Process, Queue
from concurrent.futures import ProcessPoolExecutor

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from frame_processor import YoloTrainedModelFrameProcessor


if __name__ == '__main__':
    calibration_source = './data/mp4s/roughly_880_cans.mp4'
    test_source = './data/mp4s/roughly_880_cans.mp4'
    weights = './weights/best_600_epoch_new_dataset.pt'
    img_size = 640
    conf_thresh = 0.25
    iou_thresh = 0.45
    # splits = [0.0, 0.2, 0.5, 1.0]
    fps = 20
    min_splits = 3
    processes = 10

    network_params = (weights, img_size, conf_thresh, iou_thresh)
    print('Initializing trained network...')
    frame_processor = YoloTrainedModelFrameProcessor(weights, img_size, conf_thresh, iou_thresh)

    print('Preloading calibration video into memory to simulate live-streaming.')
    cap = cv2.VideoCapture(calibration_source)
    success = True
    cal_frames = []
    count = 0
    while success:
        success, img = cap.read()
        if not success:
            break
        else:
            if fps == 60:
                cal_frames.append(img)
            if fps == 20:
                if (count % 20) == 0:
                    cal_frames.append(img)
        count += 1

    print('Preloading test video into memory to simulate live-streaming.')
    cap = cv2.VideoCapture(test_source)
    success = True
    test_frames = []
    count = 0
    while success:
        success, img = cap.read()
        if not success:
            break
        else:
            if fps == 60:
                test_frames.append(img)
            if fps == 20:
                if (count % 20) == 0:
                    test_frames.append(img)
        count += 1


# This function assumes there are no dropped frames. In the future the time data for each frame will be important
def single_can_calibrate_frame_splits():
    """The can getting detected at the edge of the frame makes the center follow a non-ballistic trajectory,
    so bounds have been established experimentally.
    """
    lower_bound = 0.2
    upper_bound = 0.8
    frame_count = 0
    cap = cv2.VideoCapture(calibration_source)
    success = True

    y_axis = []
    times = []
    y_current = 0
    tracking = False
    while success:
        success, img0 = cap.read()
        if not success:
            break

        detections = frame_processor.detect_cans(img0)
        if len(detections) == 1:
            tracking = True
            y_val = detections[0][1]
            if lower_bound < y_val < upper_bound:
                y_axis.append(y_val)
                times.append(frame_count*1/fps)
                y_current = y_val
        else:
            if lower_bound < y_current < upper_bound and tracking:  # It would get here if it loses tracking for whatever reason
                print('calibration failed. A detection failed around y =', y_current)
                return
        frame_count += 1

    plt.ylim([-1, 0])
    plt.scatter(times, y_axis)
    plt.savefig("falling_can.png")
    print(y_axis)

    return y_axis


def multi_can_calibrate_frame_splits():
    """
    Similar to single_can_calibrate_frame_splits() but looks for a gap in the stream of cans and then tracks the leading
    can.
    """
    lower_bound = 0.3
    upper_trigger = 0.65
    upper_bound = 0.75
    frame_count = 0
    cap = cv2.VideoCapture(calibration_source)
    success = True

    y_axis = []
    y_plot = []
    times = []
    y_previous = 0
    y_current = 0
    start_searching = False
    tracking = False
    while success:
        success, img0 = cap.read()
        if not success:
            break

        detections = frame_processor.detect_cans(img0)
        if not start_searching:
            for detection in detections:  # This loop is trying to find a frame where no cans are in the ROI
                start_searching = True
                if lower_bound < detection[1] < upper_bound:
                    start_searching = False
            frame_count += 1
            if start_searching:
                print('Found suitable gap in can stream ending at frame', frame_count)
            continue

        if start_searching:  # Once a frame with no cans in the ROI is found, this will execute
            for detection in detections:  # By the end of this loop the y_val for the leading can will be found
                y_val = detection[1]
                if y_val > y_current:
                    y_current = y_val
                    tracking = True

            if y_current <= y_previous:  # Leading can somehow reversed direction, so tracking must have been lost.
                print('tracking was lost on lead can at frame', frame_count, '. Searching for new gap in can stream.')
                y_axis = []
                times = []
                y_previous = 0
                y_current = 0
                start_searching = False
                tracking = False
                frame_count += 1
                continue

            if tracking:
                if lower_bound < y_current < upper_bound:
                    y_axis.append(y_current)
                    y_plot.append(-y_current)
                    times.append(frame_count * 1 / fps)
                    if y_current > upper_trigger:
                        print('Calibration complete. Can successfully tracked to frame ', frame_count)
                        break
        frame_count += 1

    if len(y_axis) < min_splits:
        print('not enough splits were tracked. Either decrease the minimum number of splits, increase the frame rate, or try another calibration source.')
        return []
    plt.ylim([-1, 0])
    plt.scatter(times, y_plot)
    plt.savefig("tracked_from_falling_cans.png")

    return y_axis


def count_cans(splits):
    if len(test_frames) > 0:
        total_cans = 0
        n_frame_avg = len(
            splits) - 1  # Number of frames being averaged. This should be equal to number of regions per frame
        detections_q = queue.Queue(maxsize=n_frame_avg)  # Initialize frame queue (FIFO) with max size of n_frame_avg

        count = 0
        t1 = time.time_ns()
        for img0 in test_frames:
            count += 1
            detections = frame_processor.detect_cans(img0)
            # detections = [(1, 1, 1, 1), (1, 1, 1, 1)]
            # Count Cans
            detections = torch.Tensor(detections)  # Detections of one frame

            if detections_q.full():
                # If queue is full, pop the oldest detections and add the latest detections (first in first out)
                _ = detections_q.get()
                detections_q.put(detections)
            else:
                detections_q.put(detections)

            if detections_q.full():
                # Only starts counting when the queue is full
                # Region parameters (replace with experimental values)
                # r_bottom, r_top = 0.0, 0.2
                # b_bottom, b_top = 0.2, 0.5
                # g_bottom, g_top = 0.5, 1.0
                n_detections = []
                for i, frame_detections in enumerate(list(detections_q.queue)):
                    # total_n_cans_in_frame_i = frame_detections.shape[0]
                    # Number of detections that are in certain region
                    if frame_detections.shape[0] == 0:
                        n_detections.append(0)
                    else:
                        n_detections_region_i = torch.sum(
                            np.logical_and(splits[i] <= frame_detections[:, 1], frame_detections[:, 1] < splits[i + 1]))
                        n_detections.append(n_detections_region_i)

                # Average (switch to other rounding methods if desired)
                # num_cans = np.average(n_detections)
                num_cans = int(np.round(np.average(n_detections)))
                total_cans += num_cans
        t2 = time.time_ns()
        ave_frame_time = (t2 - t1) / (count + 1) / 1e9
        processed_fps = 1 / ave_frame_time
        print('total frames:', count + 1)
        print('Average time per frame:', ave_frame_time)
        print('average frame rate:', processed_fps)
        return total_cans

    else:
        total_cans = 0
        n_frame_avg = len(splits) - 1  # Number of frames being averaged. This should be equal to number of regions per frame
        detections_q = queue.Queue(maxsize=n_frame_avg)  # Initialize frame queue (FIFO) with max size of n_frame_avg

        cap = cv2.VideoCapture(calibration_source)
        success = True
        count = 0
        t1 = time.time_ns()
        while success:
            count += 1
            success, img0 = cap.read()
            if not success:
                break

            detections = frame_processor.detect_cans(img0)
            # detections = [(1, 1, 1, 1), (1, 1, 1, 1)]
            # Count Cans
            detections = torch.Tensor(detections)  # Detections of one frame

            if detections_q.full():
                # If queue is full, pop the oldest detections and add the latest detections (first in first out)
                _ = detections_q.get()
                detections_q.put(detections)
            else:
                detections_q.put(detections)

            if detections_q.full():
                # Only starts counting when the queue is full
                # Region parameters (replace with experimental values)
                # r_bottom, r_top = 0.0, 0.2
                # b_bottom, b_top = 0.2, 0.5
                # g_bottom, g_top = 0.5, 1.0
                n_detections = []
                for i, frame_detections in enumerate(list(detections_q.queue)):
                    # total_n_cans_in_frame_i = frame_detections.shape[0]
                    # Number of detections that are in certain region
                    if frame_detections.shape[0] == 0:
                        n_detections.append(0)
                    else:
                        n_detections_region_i = torch.sum(
                            np.logical_and(splits[i] <= frame_detections[:, 1], frame_detections[:, 1] < splits[i + 1]))
                        n_detections.append(n_detections_region_i)

                # Average (switch to other rounding methods if desired)
                # num_cans = np.average(n_detections)
                num_cans = int(np.round(np.average(n_detections)))
                total_cans += num_cans
        t2 = time.time_ns()
        ave_frame_time = (t2-t1)/(count+1)/1e9
        processed_fps = 1/ave_frame_time
        print('total frames:', count+1)
        print('Average time per frame:', ave_frame_time)
        print('average frame rate:', processed_fps)
        return total_cans


def count_cans_multi_preloaded(splits):
    count = 0
    total_cans = 0
    n_frame_avg = len(splits) - 1  # Number of frames being averaged. This should be equal to number of regions per frame
    detections_q = queue.Queue(maxsize=n_frame_avg)  # Initialize frame queue (FIFO) with max size of n_frame_avg

    t1 = time.time_ns()
    for img0 in test_frames:
        count += 1

        detections = frame_processor.detect_cans(img0)
        # detections = [(1, 1, 1, 1), (1, 1, 1, 1)]
        # Count Cans
        detections = torch.Tensor(detections)  # Detections of one frame

        if detections_q.full():
            # If queue is full, pop the oldest detections and add the latest detections (first in first out)
            _ = detections_q.get()
            detections_q.put(detections)
        else:
            detections_q.put(detections)

        if detections_q.full():
            # Only starts counting when the queue is full
            # Region parameters (replace with experimental values)
            # r_bottom, r_top = 0.0, 0.2
            # b_bottom, b_top = 0.2, 0.5
            # g_bottom, g_top = 0.5, 1.0
            n_detections = []
            for i, frame_detections in enumerate(list(detections_q.queue)):
                # total_n_cans_in_frame_i = frame_detections.shape[0]
                # Number of detections that are in certain region
                if frame_detections.shape[0] == 0:
                    n_detections.append(0)
                else:
                    n_detections_region_i = torch.sum(
                        np.logical_and(splits[i] <= frame_detections[:, 1], frame_detections[:, 1] < splits[i + 1]))
                    n_detections.append(n_detections_region_i)

            # Average (switch to other rounding methods if desired)
            # num_cans = np.average(n_detections)
            num_cans = int(np.round(np.average(n_detections)))
            total_cans += num_cans
    t2 = time.time_ns()
    ave_frame_time = (t2 - t1) / (count + 1) / 1e9
    processed_fps = 1 / ave_frame_time
    print('total frames:', count + 1)
    print('Average time per frame:', ave_frame_time)
    print('average frame rate:', processed_fps)
    return total_cans


def simple_detect(images, ids, params):
    sub_frame_processor = YoloTrainedModelFrameProcessor(*params)
    detections_list = []
    ret_val = None
    t1 = time.time_ns()
    for image in images:
        detections = sub_frame_processor.detect_cans(image)
        detections_list.append(detections)
        ret_val = (detections_list, ids)
    t2 = time.time_ns()
    print('Time:', (t2 - t1) / 1e9)
    return ret_val


def multi_test_detect():
    multi = True
    if multi:
        test__frame_ids = list(range(len(test_frames)))
        split_test_frames = np.array_split(test_frames, processes)
        split_frame_ids = np.array_split(test__frame_ids, processes)
        t1 = time.time_ns()
        results = []
        with ProcessPoolExecutor() as executor:

            process_list = []
            for i in range(processes):
                f = executor.submit(simple_detect, split_test_frames[i], split_frame_ids[i], network_params)
                process_list.append(f)
                print('Submitted process', i+1)

            for process in process_list:
                results.append(process.result())

        t2 = time.time_ns()
        print('Total time:', (t2 - t1) / 1e9)
    else:
        t1 = time.time_ns()
        detections_list = []
        for img0 in test_frames:
            detections = frame_processor.detect_cans(img0)
            detections_list += detections
        print('Detections:', detections_list)
        t2 = time.time_ns()
        print('Time:', (t2 - t1) / 1e9)


def main():
    print('Starting Calibration...')
    splits = multi_can_calibrate_frame_splits()
    print('splits:', splits)
    if len(splits) < min_splits:
        return
    print('Counting cans...')
    multi_test_detect()
    # total_cans = count_cans(splits)
    # print('total cans', total_cans)


if __name__ == '__main__':
    main()


# python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/
