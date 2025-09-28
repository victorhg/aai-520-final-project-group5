class BaseWorker:
    def __init__(self):
        pass

    def execute(self, *inputs):
        raise NotImplementedError("This method should be overridden by subclasses.")