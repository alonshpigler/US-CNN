
import time

class Timer:

    def __init__(self, operation=''):

        self.operation = operation
        self.start_time = None
        self.end_time = None

    def start(self):

        self.__log_start()
        self.start_time = time.time()
        return self

    def end(self):
        self.end_time = time.time()
        self.__log_end()

    def __log_start(self):
        print(self.operation + " Start Measure Time")

    def __log_end(self):
        print(self.operation + " Elapsed Time: " + str(self.end_time - self.start_time))
