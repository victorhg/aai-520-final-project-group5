class BaseWorker:
    def execute(self, *inputs):
        """Execute the worker's main function. To be overridden by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    # def send_to(self, other_agent, message):
    #     # Module 7 lab uses this. But I think we shouldn't. Just adding this here to say that.
    #     # I think our orchestrator should handle all the communication between workers.
    #     pass
    