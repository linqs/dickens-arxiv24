import os

from models.symbolic_result_formulator.base_symbolic_result_formulator import BaseSymbolicResultFormulator
from utils import SYMBOLIC_REASONER_DIR
from utils import load_psl_file


class NeuPSLResultFormulator(BaseSymbolicResultFormulator):
    def __init__(self, arguments):
        super().__init__(arguments)

        self.inferred_predicates_dir = os.path.join(SYMBOLIC_REASONER_DIR, arguments.symbolic_reasoner_name, "cli", "inferred-predicates")

    def formulate_symbolic_results(self, symbolic_program_formulator):
        if not os.path.exists(self.inferred_predicates_dir):
            return ""

        if symbolic_program_formulator.query is not None:
            inferred_predicates_dict = self._query_inferred_predicates(symbolic_program_formulator.query)
        else:
            inferred_predicates_dict = self._max_class_inferred_predicates()

        return self._format_inferred_predicates(inferred_predicates_dict)

    def _format_inferred_predicates(self, inferred_predicates_dict):
        formatted_string = ""
        for predicate_name in sorted(inferred_predicates_dict.keys()):
            result_string = ""
            for value in inferred_predicates_dict[predicate_name]:
                result_string += predicate_name + "(" + ",".join(value[:-1]) + ") = " + str(value[-1]) + "\n"
            formatted_string += result_string + "\n"

        return formatted_string

    def _query_inferred_predicates(self, query):
        results_dict = {}
        max_value = 0
        max_index = -1
        for filename in os.listdir(self.inferred_predicates_dir):
            results_dict[filename[:-4].lower()] = []

            file_results = load_psl_file(os.path.join(self.inferred_predicates_dir, filename))
            for example in file_results:
                example = [variable.replace(" ", "") for variable in example]
                if (filename[:-4].lower(), tuple(example[:-1])) not in query:
                    continue
                if float(example[-1]) > max_value:
                    max_value = float(example[-1])
                    max_index = len(results_dict[filename[:-4].lower()])
                results_dict[filename[:-4].lower()].append(list(example[:-1]) + ["0"])

            if len(results_dict[filename[:-4].lower()]) > 0:
                results_dict[filename[:-4].lower()][max_index][-1] = "1"

        return results_dict

    def _max_class_inferred_predicates(self):
        results_dict = {}
        for filename in os.listdir(self.inferred_predicates_dir):
            file_results = load_psl_file(os.path.join(self.inferred_predicates_dir, filename))
            file_results_dict = {}
            for example in file_results:
                if tuple(example[:-2]) not in file_results_dict:
                    file_results_dict[tuple(example[:-2])] = []
                file_results_dict[tuple(example[:-2])].append((example[-2], float(example[-1])))

            unsorted_results = []
            for key in file_results_dict:
                max_index = -1
                for value_index in range(len(file_results_dict[key])):
                    if max_index == -1 or file_results_dict[key][max_index][-1] < file_results_dict[key][value_index][-1]:
                        max_index = value_index
                result = list(key)
                result.extend(list(file_results_dict[key][max_index])[:-1])
                unsorted_results.append(result)

            results_dict[filename[:-4]] = sorted(unsorted_results, key=lambda x :(x[1]))

        return results_dict
