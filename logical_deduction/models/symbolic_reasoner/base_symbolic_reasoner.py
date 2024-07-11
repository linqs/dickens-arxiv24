import abc


class BaseSymbolicReasoner(abc.ABC):
    def __init__(self, arguments):
        self.arguments = arguments

    @abc.abstractmethod
    def run_symbolic_reasoner(self, problem):
        pass
