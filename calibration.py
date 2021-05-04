def multi_can_calibrate_frame_splits(frame_processor, video, lower_bound, upper_trigger, upper_bound, min_splits):
    """
    Similar to single_can_calibrate_frame_splits() but looks for a gap in the stream of cans and then tracks the leading
    can.
    """
    frame_count = 0
    y_axis = []
    y_previous = 0
    y_current = 0
    start_searching = False
    tracking = False
    for img0 in video:
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

            if y_current <= y_previous or y_current > upper_bound:  # Leading can somehow reversed direction or skipped the upper bound, so tracking must have been lost.
                print('tracking was lost on lead can at frame', frame_count, '. Searching for new gap in can stream.')
                y_axis = []
                y_previous = 0
                y_current = 0
                start_searching = False
                tracking = False
                frame_count += 1
                continue

            if tracking:
                if lower_bound < y_current < upper_bound:
                    y_axis.append(y_current)
                    if y_current > upper_trigger:
                        if len(y_axis) < min_splits:
                            print('not enough splits were tracked. Continuing search...')
                            y_axis = []
                            y_previous = 0
                            y_current = 0
                            start_searching = False
                            tracking = False
                            frame_count += 1
                            continue
                        else:
                            print('Calibration complete. Can successfully tracked to frame ', frame_count)
                            break
        frame_count += 1
    if len(y_axis) < min_splits:
        print('No viable splits could be found for the given footage. Change calibration parameters or footage.')
    return y_axis
