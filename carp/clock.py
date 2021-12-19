import time

def get_millis():
    return int(round(time.time() * 1000))

class Clock:
    def __init__(self):
        self.timer = get_millis()
    def hit(self):
        new_time = get_millis()
        diff = new_time - self.timer
        self.timer = new_time
        return diff
