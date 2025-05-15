# processBienBao.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../..")

from src.templates.workerprocess import WorkerProcess
from src.bien_bao.threads.threadTrafficsign import threadTrafficsign

class processTrafficsign(WorkerProcess):
    def __init__(self, queueList, logger, debugging=False):
        self.logger = logger
        self.debugging = debugging
        super(processTrafficsign, self).__init__(queueList)

    def run(self):
        super(processTrafficsign, self).run()

    def _init_threads(self):
        threadTrafficsign = threadTrafficsign(
            self.queuesList, self.logger, self.debugging
        )
        self.threads.append(threadTrafficsign)


# ======== Only runs when you call this script directly ========
if __name__ == "__main__":
    from multiprocessing import Queue
    import time
    import logging

    queueList = {
        "Critical": Queue(),
        "Warning": Queue(),
        "General": Queue(),
        "Config": Queue(),
    }

    logging = logging.getLogger()
    process = processTrafficsign(queueList, logging, debugging=True)
    process.daemon = True
    process.start()

    time.sleep(10)  # let it run for a while
    process.stop()

