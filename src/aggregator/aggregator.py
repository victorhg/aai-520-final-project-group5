from worker.base_worker import BaseWorker

class Aggregator(BaseWorker):
    def execute(self, *inputs) -> str:
        return "Sample data from aggregator worker"
