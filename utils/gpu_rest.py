import time

class GPURest():
    def __init__(self, rest_duration=300):
        self.rest_duration = rest_duration

    def rest(self):
        print(f"Resting for {self.rest_duration} seconds.")
        time.sleep(self.rest_duration)
        print("Resuming training...")