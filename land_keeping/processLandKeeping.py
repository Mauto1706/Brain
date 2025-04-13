
from src.templates.workerprocess import WorkerProcess
from src.hardware.lanekeeping.threads.threadLandKeeping import ThreadLandKeeping  # ??m b?o b?n ?ã có thread `ThreadLandKeeping`

class processLandkeeping(WorkerProcess):
    """This process handles land keeping (lane detection and steering control)."""
    
    def __init__(self, queueList, logging, debugging=False):
        """Initialize the land keeping process."""
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        super(processLandkeeping, self).__init__(self.queuesList)
    
    def run(self):
        """Start the land keeping process (thread)."""
        super(processLandkeeping, self).run()

    def _init_threads(self):
        """Initialize the land keeping thread and add it to the list of threads."""
        # Khoi tao thread xu lý gi làn (land keeping)
        landKeepingThread = ThreadLandKeeping(self.queuesList, self.logging, self.debugging)
        self.threads.append(landKeepingThread)
