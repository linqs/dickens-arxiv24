#!/usr/bin/env python3

import os
import re
import sys

import pandas as pd

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', '..'))

import utils
import visual_sudoku_solving.experiments.experiment_utils as experiment_utils

RESULTS_DIR = os.path.join(THIS_DIR, '..', '..', 'results', 'inference')
INFERRED_PREDICATES_DIR_NAME = 'inferred-predicates'
LOG_FILENAME = 'out.txt'


def tabulate_results(results):
    tables = {}
    for experiment, experiment_result in sorted(results.items()):
        tables[experiment] = {}
        for model, model_result in sorted(experiment_result.items()):
            tables[experiment][model] = {col_name: [] for col_name in model_result['header']}
            for row in model_result['rows']:
                for i, col_name in enumerate(model_result['header']):
                    tables[experiment][model][col_name].append(row[i])

    return tables


def parse_log(log_path):
    results = []
    with open(log_path, 'r') as file:
        for line in file:
            if 'Evaluation results: Evaluator: CategoricalEvaluator, Predicate: IMAGEDIGIT' in line:
                match = re.search(r': ([\d\.]+)', line)
                results.append(float(match.group(1)))

            if 'Evaluation results: Evaluator: CategoricalEvaluator, Predicate: NEURALCLASSIFIER' in line:
                match = re.search(r': ([\d\.]+)', line)
                results.append(float(match.group(1)))

    return results


def main():
    results = {}
    for experiment in sorted(os.listdir(RESULTS_DIR)):
        results[experiment] = {setting: dict() for setting in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment)))}
        for model in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment))):
            results[experiment][model] = {'header': [], 'rows': []}
            log_paths = utils.get_log_paths(os.path.join(RESULTS_DIR, experiment, model), LOG_FILENAME)
            for log_path in log_paths:
                parts = os.path.dirname(
                    log_path.split("{}/{}/".format(experiment, model))[1]).split("/")
                if len(results[experiment][model]['rows']) == 0:
                    results[experiment][model]['header'] = [row.split("::")[0] for row in parts]

                    # Assuming that the evaluator logs IMAGEDIGIT then NEURALCLASSIFIER categorical accuracy in that order.
                    results[experiment][model]['header'].append('IMAGEDIGIT_Categorical_Accuracy')
                    results[experiment][model]['header'].append('NEURALCLASSIFIER_Categorical_Accuracy')

                    results[experiment][model]['header'].append('IMAGEDIGIT_PUZZLE_CONSISTENCY')
                    results[experiment][model]['header'].append('NEURALCLASSIFIER_PUZZLE_CONSISTENCY')

                results[experiment][model]['rows'].append([row.split("::")[1] for row in parts])

                for log_result in parse_log(log_path):
                    results[experiment][model]['rows'][-1].append(log_result)

                results[experiment][model]['rows'][-1].append(
                    experiment_utils.get_puzzle_consistency(
                        os.path.join(os.path.dirname(log_path), INFERRED_PREDICATES_DIR_NAME, "IMAGEDIGIT.txt"))
                )

                results[experiment][model]['rows'][-1].append(
                    experiment_utils.get_puzzle_consistency(
                        os.path.join(os.path.dirname(log_path), INFERRED_PREDICATES_DIR_NAME, "NEURALCLASSIFIER.txt"))
                )

    tabulated_results = tabulate_results(results)
    for experiment, experiment_result in sorted(tabulated_results.items()):
        for model, model_result in sorted(experiment_result.items()):
            result_dataframe = pd.DataFrame(tabulated_results[experiment][model])
            if result_dataframe.empty:
                continue
            result_dataframe.set_index('split', inplace=True)
            result_dataframe.to_csv(os.path.join(RESULTS_DIR, experiment, model, 'results.csv'))


def _load_args(args):
    executable = args.pop(0)
    if len(args) != 0 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3 %s" % (executable,), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    _load_args(sys.argv)
    main()
