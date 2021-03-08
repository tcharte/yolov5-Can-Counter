import time

t1 = time.time_ns()
time.sleep(15)
t2 = time.time_ns()
print(t2, t1, (t2-t1)/1e9)
