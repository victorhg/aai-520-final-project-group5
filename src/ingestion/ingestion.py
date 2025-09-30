from src.worker.base_worker import BaseWorker

class Ingestion(BaseWorker):
    def execute(self, *inputs) -> str:
        return "Sample data from ingestion worker"
