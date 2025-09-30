class BaseWorker:
    def execute(self, *inputs):
        """Execute the worker's main function. To be overridden by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")
