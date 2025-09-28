class BaseWorker:
    def __init__(self, name, role, model="gpt-3.5-turbo"):
        self.name = name
        self.role = role
        self.model = model # if a worker is not an agent, this can be None

    def execute(self, *inputs):
        """Execute the worker's main function. To be overridden by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    # def send_to(self, other_agent, message):
    #     # Module 7 lab uses this. But I think we shouldn't. Just adding this here to say that.
    #     # I think our orchestrator should handle all the communication between workers.
    #     pass
    