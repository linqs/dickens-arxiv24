import abc


class BaseSymbolicProblemFormulator(abc.ABC):
    def __init__(self, arguments):
        self.arguments = arguments

    @abc.abstractmethod
    def formulate_symbolic_problem(self, response):
        pass
