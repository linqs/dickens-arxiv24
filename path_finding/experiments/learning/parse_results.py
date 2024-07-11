#!/usr/bin/env python3

import os
import re
import sys

import pandas as pd

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..', '..'))

import utils
import path_finding.experiments.experiment_utils as experiment_utils

BASE_DATA_DIR = os.path.join(THIS_DIR, '..', '..', 'data')
DATA_DIR_FORMAT = os.path.join(BASE_DATA_DIR, '{}', '{}', '{}')
INFERRED_PREDICATES_DIR_NAME = 'inferred-predicates'
LOG_FILENAME = 'out.txt'
RESULTS_DIR = os.path.join(THIS_DIR, '..', '..', 'results', 'learning')


def tabulate_results(results):
    tables = {}
    for experiment, experiment_result in sorted(results.items()):
        tables[experiment] = {}
        for learning_method, learning_method_result in sorted(experiment_result.items()):
            tables[experiment][learning_method] = {}
            for model, model_result in sorted(learning_method_result.items()):
                tables[experiment][learning_method][model] = {}
                for learning_method_variant, learning_method_variant_result in sorted(model_result.items()):
                    tables[experiment][learning_method][model][learning_method_variant] = {col_name: [] for col_name in learning_method_variant_result['header']}
                    for row in learning_method_variant_result['rows']:
                        for i, col_name in enumerate(learning_method_variant_result['header']):
                            if i < len(row):
                                tables[experiment][learning_method][model][learning_method_variant][col_name].append(row[i])
                            else:
                                tables[experiment][learning_method][model][learning_method_variant][col_name].append(None)

    return tables


def parse_experiment_result(log_path):
    results = []
    with open(log_path, 'r') as file:
        for line in file:
            if 'Final MAP State Validation Evaluation Metric:' in line:
                match = re.search(r': ([\d\.]+)', line)
                results.append(float(match.group(1)))

            if 'Evaluation results: Evaluator: DiscreteEvaluator, Predicate: ONPATH' in line:
                match = re.search(r'F1: ([\d\.]+)', line)
                results.append(float(match.group(1)))

    return results


def main():
    results = {}
    for experiment in sorted(os.listdir(RESULTS_DIR)):
        results[experiment] = {setting: dict() for setting in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment)))}
        for learning_method in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment))):
            results[experiment][learning_method] = {dataset: dict() for dataset in
                                                 sorted(os.listdir(os.path.join(RESULTS_DIR, experiment, learning_method)))}
            for model in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment, learning_method))):
                results[experiment][learning_method][model] = {learning_method_variant: dict() for learning_method_variant in sorted(
                    os.listdir(os.path.join(RESULTS_DIR, experiment, learning_method, model)))}
                for learning_method_variant in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment, learning_method, model))):
                    results[experiment][learning_method][model][learning_method_variant] = {'header': [], 'rows': []}
                    log_paths = utils.get_log_paths(os.path.join(RESULTS_DIR, experiment, learning_method, model, learning_method_variant), LOG_FILENAME)
                    for log_path in log_paths:
                        parts = os.path.dirname(
                            log_path.split("{}/{}/{}/{}/".format(experiment, learning_method, model, learning_method_variant))[1]).split("/")
                        if len(results[experiment][learning_method][model][learning_method_variant]['rows']) == 0:
                            results[experiment][learning_method][model][learning_method_variant]['header'] = [
                                row.split("::")[0] for row in parts
                            ]

                            # Assuming that the evaluator logs results in the following order.
                            if learning_method in ['none', 'constraint_satisfaction']:
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('ONPATH_Test_F1')
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('Test_Neural_Path_Continuous_Consistency')
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('Test_Neural_EBM_Path_Min_Cost_Consistency')
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('Test_NeSy_EBM_Path_Continuous_Consistency')
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('Test_NeSy_EBM_Path_Min_Cost_Consistency')
                            else:
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('ONPATH_Validation_F1')
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('ONPATH_Test_F1')
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('Test_Neural_Path_Continuous_Consistency')
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('Test_Neural_EBM_Path_Min_Cost_Consistency')
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('Test_NeSy_EBM_Path_Continuous_Consistency')
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('Test_NeSy_EBM_Path_Min_Cost_Consistency')

                        results[experiment][learning_method][model][learning_method_variant]['rows'].append(
                            [row.split("::")[1] for row in parts])

                        for log_result in parse_experiment_result(log_path):
                            results[experiment][learning_method][model][learning_method_variant]['rows'][-1].append(log_result)

                        continous_consistency, min_cost_consistency = experiment_utils.get_path_consistency(
                            os.path.join(os.path.dirname(log_path), INFERRED_PREDICATES_DIR_NAME, "NEURALPATHFINDER.txt"),
                            os.path.join(DATA_DIR_FORMAT.format("warcraft-map", *parts), "path-truth-test.txt"),
                            os.path.join(DATA_DIR_FORMAT.format("warcraft-map", *parts), "map-costs-truth-test.txt"),
                            xy_inferred_format=False
                        )

                        results[experiment][learning_method][model][learning_method_variant]['rows'][-1].append(continous_consistency)
                        results[experiment][learning_method][model][learning_method_variant]['rows'][-1].append(min_cost_consistency)

                        continous_consistency, min_cost_consistency = experiment_utils.get_path_consistency(
                            os.path.join(os.path.dirname(log_path), INFERRED_PREDICATES_DIR_NAME, "ONPATH.txt"),
                            os.path.join(DATA_DIR_FORMAT.format("warcraft-map", *parts), "path-truth-test.txt"),
                            os.path.join(DATA_DIR_FORMAT.format("warcraft-map", *parts), "map-costs-truth-test.txt")
                        )

                        results[experiment][learning_method][model][learning_method_variant]['rows'][-1].append(continous_consistency)
                        results[experiment][learning_method][model][learning_method_variant]['rows'][-1].append(min_cost_consistency)

    tabulated_results = tabulate_results(results)
    for experiment, experiment_result in sorted(tabulated_results.items()):
        for learning_method, learning_method_result in sorted(experiment_result.items()):
            for model, model_result in sorted(learning_method_result.items()):
                for learning_method_variant in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment, learning_method, model))):
                    result_dataframe = pd.DataFrame(tabulated_results[experiment][learning_method][model][learning_method_variant])
                    if result_dataframe.empty:
                        continue
                    result_dataframe.set_index('split', inplace=True)
                    result_dataframe.to_csv(os.path.join(RESULTS_DIR, experiment, learning_method, model, learning_method_variant, 'results.csv'))


def _load_args(args):
    executable = args.pop(0)
    if len(args) != 0 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3 %s" % (executable,), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    _load_args(sys.argv)
    main()
