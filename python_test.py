import multiprocessing
import queue


quwu = multiprocessing.Queue(maxsize=100000)
try:
    yeet = quwu.get(block=False)
except queue.Empty:
    print('uh oh empty')