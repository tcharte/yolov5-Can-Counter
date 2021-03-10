def count_cans(splits, detections_queue, count_queue):
    import queue

    import numpy as np
    import torch

    detections_list = []
    n_frame_avg = len(
        splits) - 1  # Number of frames being averaged. This should be equal to number of regions per frame
    detections_q = queue.Queue(maxsize=n_frame_avg)  # Initialize frame queue (FIFO) with max size of n_frame_avg
    print('Ready to count cans!')
    total_cans = 0
    while True:
        detections, frame_id = detections_queue.get()  # All cans detected in one frame
        # detections = [(1, 1, 1, 1), (1, 1, 1, 1)]
        detections = torch.Tensor(detections)  # Converting to tensor for ease of calculation

        detections_list.append(detections)

        if len(detections_list) == n_frame_avg:
            # Only starts counting when the queue is full
            n_detections = []
            for i, frame_detections in enumerate(detections_list):
                # total_n_cans_in_frame_i = frame_detections.shape[0]
                # Number of detections that are in certain region
                if frame_detections.shape[0] == 0:
                    n_detections.append(0)
                else:
                    n_detections_region_i = torch.sum(
                        np.logical_and(splits[i] <= frame_detections[:, 1], frame_detections[:, 1] < splits[i + 1]))
                    n_detections.append(n_detections_region_i)

            del detections_list[0]
            # Average (switch to other rounding methods if desired)
            # num_cans = np.average(n_detections)
            num_cans = int(np.round(np.average(n_detections)))
            total_cans += num_cans
            if count_queue.full():
                _ = count_queue.get()
                count_queue.put((total_cans, frame_id))
            else:
                count_queue.put((total_cans, frame_id))
