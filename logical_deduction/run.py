import argparse
import logging
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger

from models.problem_formulator import ProblemFormulator
from models.prompts.prompt_creator import PromptCreator
from models.result_interpreter import ResultInterpreter
from models.symbolic_problem_formulator.neupsl.neupsl_problem_formulator import NeuPSLProblemFormulator
from models.symbolic_reasoner.neupsl.neupsl_reasoner import NeuPSLReasoner
from models.symbolic_result_formulator.neupsl.neupsl_result_formulator import NeuPSLResultFormulator
from utils import DATA_DIR
from utils import DEFAULT_LOGGING_LEVEL
from utils import PROMPTS_DIR
from utils import RESULTS_DIR
from utils import load_json_file
from utils import load_txt_file
from utils import write_json_file


def run_symbolic_llm(arguments):
    dataset = load_json_file(os.path.join(arguments.data_dir, arguments.dataset_name, arguments.dataset_filename + ".json"))
    problem_formulator = ProblemFormulator(arguments)
    result_interpreter = ResultInterpreter(arguments)
    prompt_creator = PromptCreator(arguments)
    problem_formulator_prompt_template = load_txt_file(os.path.join(arguments.prompts_dir, arguments.symbolic_reasoner_name, arguments.dataset_name, "problem-formulator.txt"))
    results_formulator_prompt_template = load_txt_file(os.path.join(arguments.prompts_dir, arguments.symbolic_reasoner_name, arguments.dataset_name, "result-interpreter.txt"))
    if arguments.symbolic_reasoner_name == "neupsl":
        symbolic_program_generator = NeuPSLProblemFormulator(arguments)
        symbolic_reasoner = NeuPSLReasoner(arguments)
        symbolic_results_formulator = NeuPSLResultFormulator(arguments)
    else:
        raise NotImplementedError

    results = {'correct': 0,
               'incorrect': 0,
               'accuracy': 0.0,
               'symbolic_reasoner_empty': 0}

    for example in tqdm(dataset):
        problem_formulator_prompt = prompt_creator.create_problem_formulator_prompt(example, problem_formulator_prompt_template)
        problem_formulator_response = problem_formulator.generate_problem_formulation(example, problem_formulator_prompt)
        example['problem_formulation_response'] = problem_formulator_response

        symbolic_program_formulator_response = symbolic_program_generator.formulate_symbolic_problem(problem_formulator_response)
        example['symbolic_program_formulation_response'] = symbolic_program_formulator_response

        symbolic_reasoner.run_symbolic_reasoner(symbolic_program_formulator_response)

        symbolic_results_formulator_response = symbolic_results_formulator.formulate_symbolic_results(symbolic_program_generator)
        example['symbolic_results_formulation_response'] = symbolic_results_formulator_response

        if symbolic_results_formulator_response == "":
            results["symbolic_reasoner_empty"] += 1
            example['result_interpretation_response'] = example['context']

        result_interpreter_prompt = prompt_creator.create_result_interpreter_prompt(example, results_formulator_prompt_template, symbolic_results_formulator_response)
        result_interpreter_response = result_interpreter.generate_result_interpretation(example, result_interpreter_prompt)
        example['result_interpretation_response'] = result_interpreter_response

        result = result_interpreter_response[0].upper()
        if len(result_interpreter_response.split("\n")) > 1:
            result = result_interpreter_response.split("\n")[1][0].upper()

        if  result == example['answer']:
            results['correct'] += 1
        else:
            results['incorrect'] += 1

        logging.info("Correct: %d" % (results['correct'],))
        logging.info("Incorrect: %d" % (results['incorrect'],))
        logging.info("Accuracy: %.2f" % (results['correct'] / (results['correct'] + results['incorrect']),))
        logging.info("Symbolic Reasoner Empty: %d" % (results['symbolic_reasoner_empty'],))

        os.makedirs(os.path.join(arguments.results_dir, arguments.dataset_name, arguments.model_name, arguments.symbolic_reasoner_name), exist_ok=True)
        write_json_file(os.path.join(arguments.results_dir, arguments.dataset_name, arguments.model_name, arguments.symbolic_reasoner_name, arguments.dataset_filename + ".json"), dataset)
        write_json_file(os.path.join(arguments.results_dir, arguments.dataset_name, arguments.model_name, arguments.symbolic_reasoner_name, arguments.dataset_filename + "-results.json"), results)

        if arguments.max_examples is not None and results['correct'] + results['incorrect'] >= arguments.max_examples:
            break


def main(arguments):
    logger.initLogging(arguments.log_level)
    logging.info("Generating and running NeuPSL-LLM.")
    logging.debug("Arguments: %s" % (arguments,))

    logging.info("Running symbolic large language models on %s." % (arguments.dataset_name,))
    run_symbolic_llm(arguments)


def _load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", dest="log_level",
                        action="store", type=str, default=DEFAULT_LOGGING_LEVEL,
                        help="Logging level.", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--data-dir", dest="data_dir",
                        action="store", type=str, default=DATA_DIR,
                        help="Path to base data dir.")
    parser.add_argument("--prompts-dir", dest="prompts_dir",
                        action="store", type=str, default=PROMPTS_DIR,
                        help="Path to base prompts dir.")
    parser.add_argument("--results-dir", dest="results_dir",
                        action="store", type=str, default=RESULTS_DIR,
                        help="Path to base results dir.")
    parser.add_argument("--dataset-filename", dest="dataset_filename",
                        action="store", type=str, default="dev",
                        help="Name of dataset file.")
    parser.add_argument("--symbolic-reasoner-name", dest="symbolic_reasoner_name",
                        action="store", type=str, default="neupsl",
                        help="Name of symbolic reasoner.")
    parser.add_argument("--dataset-name", dest="dataset_name",
                        action="store", type=str, default="logical_deduction",
                        help="Name of dataset.")
    parser.add_argument("--model-name", dest="model_name",
                        action="store", type=str, default="gpt-3.5-turbo",
                        help="Name of model.")
    parser.add_argument("--stop-token", dest="stop_token",
                        action="store", type=str, default="------",
                        help="Stop token for prompt.")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens",
                        action="store", type=int, default=2048,
                        help="Max new tokens for prompt.")
    parser.add_argument("--api-key", dest="api_key",
                        action="store", type=str, default=None,
                        help="API key for OpenAI API.")
    parser.add_argument("--max-examples", dest="max_examples",
                        action="store", type=int, default=None,
                        help="Max number of examples to run.")
    parser.add_argument("--gurobi", dest="gurobi",
                        action="store_true", default=True,
                        help="Use Gurobi for solving.")

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    main(_load_args())
