import os

from models.symbolic_reasoner.base_symbolic_reasoner import BaseSymbolicReasoner
from utils import write_json_file

CLI_DIR = os.path.join(os.path.dirname(__file__), "cli")
INFERRED_PREDICATES_DIR = os.path.join(CLI_DIR, "inferred-predicates")

class NeuPSLReasoner(BaseSymbolicReasoner):
    def __init__(self, arguments):
        super().__init__(arguments)

    def run_symbolic_reasoner(self, problem):
        os.system("rm -r " + INFERRED_PREDICATES_DIR)

        write_json_file(os.path.join(CLI_DIR, "problem.json"), problem)
        os.system("cd " + CLI_DIR + " ; ./run.sh > out.txt 2> out.err")
