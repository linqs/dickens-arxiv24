#!/usr/bin/env python3

import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..', '..'))

import utils

RESULTS_DIR = os.path.join(THIS_DIR, '..', '..', 'results', 'learning')
LEARNING_CONVERGENCE_EXPERIMENTS = ["unsupervised"]
LEARNING_CONVERGENCE_PLOT_PARAMETERS_DICT = {
    "title": {
        "mnist-add1": "MNIST-Add1",
        "mnist-add2": "MNIST-Add2"
    },
    "colors": {
        "bilevel": "red",
        "energy": "blue",
        "policy_gradient": "green"
    },
    "labels": {
        "bilevel": "Bilevel",
        "energy": "Energy",
        "policy_gradient": "IndeCateR"
    }
}
LOG_FILENAME = 'out.txt'


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


def parse_log(log_path):
    results = []
    with open(log_path, 'r') as file:
        for line in file:
            if 'Final MAP State Validation Evaluation Metric:' in line:
                match = re.search(r': ([\d\.]+)', line)
                results.append(float(match.group(1)))

            if 'Evaluation results: Evaluator: CategoricalEvaluator, Predicate: IMAGEDIGIT' in line:
                match = re.search(r': ([\d\.]+)', line)
                results.append(float(match.group(1)))

            if 'Evaluation results: Evaluator: CategoricalEvaluator, Predicate: IMAGESUM' in line:
                match = re.search(r': ([\d\.]+)', line)
                results.append(float(match.group(1)))

            # if 'Evaluation results: Evaluator: CategoricalEvaluator, Predicate: NEURALCLASSIFIER' in line:
            #     match = re.search(r': ([\d\.]+)', line)
            #     results.append(float(match.group(1)))

    return results


def save_results_summary():
    results = {}
    for experiment in sorted(os.listdir(RESULTS_DIR)):
        learning_method_dirs = os.listdir(os.path.join(RESULTS_DIR, experiment))
        filtered_learning_method_dirs = []
        for learning_method_dir in learning_method_dirs:
            if os.path.isdir(os.path.join(RESULTS_DIR, experiment, learning_method_dir)):
                filtered_learning_method_dirs.append(learning_method_dir)
        results[experiment] = {setting: dict() for setting in sorted(filtered_learning_method_dirs)}
        for learning_method in sorted(filtered_learning_method_dirs):
            results[experiment][learning_method] = {
                dataset: dict() for dataset in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment, learning_method)))
            }
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
                            if learning_method == 'none':
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('IMAGEDIGIT_Test_Categorical_Accuracy')
                                # results[experiment][learning_method][model][learning_method_variant]['header'].append('NEURALCLASSIFIER_Test_Categorical_Accuracy')
                            else:
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('IMAGESUM_Validation_Categorical_Accuracy')
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('IMAGEDIGIT_Test_Categorical_Accuracy')
                                results[experiment][learning_method][model][learning_method_variant]['header'].append('IMAGESUM_Test_Categorical_Accuracy')
                                # results[experiment][learning_method][model][learning_method_variant]['header'].append('NEURALCLASSIFIER_Test_Categorical_Accuracy')

                        results[experiment][learning_method][model][learning_method_variant]['rows'].append(
                            [row.split("::")[1] for row in parts])

                        for log_result in parse_log(log_path):
                            results[experiment][learning_method][model][learning_method_variant]['rows'][-1].append(log_result)

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


def get_learning_curve(log_path):
    # Construct the learning curve from the log file.
    # The learning curve is a list of tuples, where each tuple contains a time and a validation accuracy.
    learning_curve = []
    with open(log_path, 'r') as file:
        for line in file:
            if "Current MAP State Validation Evaluation Metric:" in line:
                # The validation score is the first floating point number in the line.
                validation_score_match = re.search(r': (\d+\.\d+)', line)
                # The time is the first integer in the line.
                time_match = re.search(r'^(\d+)', line)
                learning_curve.append((int(time_match.group(1)) / 1000 / 60, float(validation_score_match.group(1))))

    return learning_curve


def plot_learning_convergence(split=0):
    learning_curve_df = pd.DataFrame(columns=['learning_method', 'model', 'time', 'validation_accuracy'])
    for experiment in LEARNING_CONVERGENCE_EXPERIMENTS:
        learning_method_dirs = os.listdir(os.path.join(RESULTS_DIR, experiment))
        filtered_learning_method_dirs = []
        for learning_method_dir in learning_method_dirs:
            if os.path.isdir(os.path.join(RESULTS_DIR, experiment, learning_method_dir)):
                filtered_learning_method_dirs.append(learning_method_dir)
        for learning_method in sorted(filtered_learning_method_dirs):
            for model in os.listdir(os.path.join(RESULTS_DIR, experiment, learning_method)):
                for learning_method_variant in os.listdir(os.path.join(RESULTS_DIR, experiment, learning_method, model)):
                    log_paths = utils.get_log_paths(os.path.join(RESULTS_DIR, experiment, learning_method, model, learning_method_variant, "split::{}".format(split)), LOG_FILENAME)
                    assert len(log_paths) == 1
                    log_path = log_paths[0]

                    learning_curve = get_learning_curve(log_path)
                    # Subtract the first time point from all the time points to make the plot more readable.
                    learning_curve = [(point[0] - learning_curve[0][0], point[1]) for point in learning_curve]
                    learning_curve_df = pd.concat([
                        learning_curve_df,
                        pd.DataFrame({
                            'learning_method': learning_method,
                            'model': model,
                            'iteration': [i for i in range(len(learning_curve))],
                            'time': [point[0] for point in learning_curve],
                            'validation_accuracy': [point[1] for point in learning_curve]
                        })
                    ])

        # Display the learning curve plots for each individual model.
        for model in learning_curve_df['model'].unique():
            plt.figure()
            plt.title(LEARNING_CONVERGENCE_PLOT_PARAMETERS_DICT['title'][model])
            for learning_method in learning_curve_df['learning_method'].unique():
                learning_method_curve = learning_curve_df[(learning_curve_df['learning_method'] == learning_method) & (learning_curve_df['model'] == model)]
                plt.plot(
                    learning_method_curve['time'], learning_method_curve['validation_accuracy'], 'o--',
                    label=LEARNING_CONVERGENCE_PLOT_PARAMETERS_DICT['labels'][learning_method],
                    color=LEARNING_CONVERGENCE_PLOT_PARAMETERS_DICT['colors'][learning_method],
                    alpha=0.7
                )
            plt.xlabel('Time (minutes)')
            plt.ylabel('Validation Accuracy')
            plt.legend()
            plt.savefig(
                os.path.join(THIS_DIR, '..', '..', 'results', 'learning', experiment, 'learning_convergence_{}.png'.format(model)),
                format='png'
            )

        # Create subplots for each model.
        fig, ax = plt.subplots(1, len(learning_curve_df['model'].unique()), figsize=(15, 5), sharey=True)
        for i, model in enumerate(learning_curve_df['model'].unique()):
            for learning_method in ["policy_gradient", "bilevel", "energy"]:
                learning_method_curve = learning_curve_df[(learning_curve_df['learning_method'] == learning_method) & (learning_curve_df['model'] == model)]
                ax[i].plot(
                    learning_method_curve['time'], learning_method_curve['validation_accuracy'], 'o--',
                    label=LEARNING_CONVERGENCE_PLOT_PARAMETERS_DICT['labels'][learning_method],
                    color=LEARNING_CONVERGENCE_PLOT_PARAMETERS_DICT['colors'][learning_method],
                    alpha=0.7
                )
            ax[i].set_title(LEARNING_CONVERGENCE_PLOT_PARAMETERS_DICT['title'][model])
            ax[i].set_xlabel('Time (minutes)')

        ax[-1].legend()
        ax[0].set_ylabel('Validation Accuracy')

        plt.savefig(
            os.path.join(THIS_DIR, '..', '..', 'results', 'learning', experiment, 'MNIST-Learning-Timing.png'),
            format='png'
        )

        # Create subplots for each model.
        fig, ax = plt.subplots(1, len(learning_curve_df['model'].unique()), figsize=(15, 5), sharey=True)
        for i, model in enumerate(learning_curve_df['model'].unique()):
            for learning_method in ["energy", "bilevel", "policy_gradient"]:
                learning_method_curve = learning_curve_df[(learning_curve_df['learning_method'] == learning_method) & (learning_curve_df['model'] == model)]
                ax[i].plot(
                    learning_method_curve['iteration'], learning_method_curve['validation_accuracy'], 'o--',
                    label=LEARNING_CONVERGENCE_PLOT_PARAMETERS_DICT['labels'][learning_method],
                    color=LEARNING_CONVERGENCE_PLOT_PARAMETERS_DICT['colors'][learning_method],
                    alpha=0.7
                )
            ax[i].set_title(LEARNING_CONVERGENCE_PLOT_PARAMETERS_DICT['title'][model])
            ax[i].set_xlabel('Epoch')

        ax[-1].legend()
        ax[0].set_ylabel('Validation Accuracy')

        plt.savefig(
            os.path.join(THIS_DIR, '..', '..', 'results', 'learning', experiment, 'MNIST-Learning-Iteration-Convergence.png'),
            format='png'
        )




def _load_args(args):
    executable = args.pop(0)
    if len(args) != 0 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3 %s" % (executable,), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    _load_args(sys.argv)
    save_results_summary()
    plot_learning_convergence()
