def sort_detections(detections_in, detections_out, splits, model_count):
    from queue import Empty
    sort_size = (len(splits) - 1) * 4*model_count
    buffer = []
    print('Detections sorting process initialized!')
    while True:
        try:
            detections = detections_in.get(block=False)
            buffer.append(detections)
            if len(buffer) == sort_size:
                buffer = sorted(buffer, key=lambda x: x[1])

                for i in range(len(buffer)):
                    if detections_out.full():
                        print('Sorted detections queue is full!. The counting algorithm is not consuming detections fast enough.')
                    else:
                        detections_out.put(buffer[i])
                    if i >= sort_size // 2:
                        del buffer[0:i+1]
                        break
            del detections
        except Empty:
            continue





