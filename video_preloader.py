import cv2
import sys


def video_preloader(source, fps):
    print('Preloading calibration video into memory to simulate live-streaming.')
    cap = cv2.VideoCapture(source)
    success = True
    frames = []
    count = 0
    while success:
        success, img = cap.read()
        y = sys.getsizeof(img)
        if not success:
            break
        else:
            if fps == 60:
                frames.append(img)
            if fps == 30:
                if (count % 2) == 0:
                    frames.append(img)
            if fps == 15:
                if (count % 4) == 0:
                    frames.append(img)
        count += 1
    return frames
