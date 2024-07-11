import logging
import os
import sys

from utils_openai import OpenAIModel


class ResultInterpreter:
    def __init__(self, arguments):
        self.openai_api = OpenAIModel(arguments.api_key, arguments.model_name, arguments.stop_token, arguments.max_new_tokens)

    def generate_result_interpretation(self, example, prompt):
        try:
            response = self.openai_api.generate(prompt)
        except:
            logging.error('Error in generating logic programs for example: ', example['id'])
            sys.exit(1)

        return response