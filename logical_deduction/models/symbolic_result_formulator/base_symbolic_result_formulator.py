import abc


class BaseSymbolicResultFormulator(abc.ABC):
    def __init__(self, arguments):
        self.arguments = arguments

    @abc.abstractmethod
    def formulate_symbolic_results(self, symbolic_program_formulator):
        pass
