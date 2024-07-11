class PromptCreator:
    def __init__(self, arguments):
        self.arguments = arguments

    def create_problem_formulator_prompt(self, example, prompt_template):
        problem = example['context']
        question = example['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()}' for choice in example['options']]).strip()

        prompt = prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question).replace('[[CHOICES]]', choices_str)
        return prompt

    def create_result_interpreter_prompt(self, example, prompt_template, inferred_response):
        problem = example['context']
        question = example['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()}' for choice in example['options']]).strip()

        prompt = prompt_template.replace('[[PROBLEM]]', problem).replace('[[INFERRED]]', inferred_response).replace('[[QUESTION]]', question).replace('[[CHOICES]]', choices_str)
        return prompt
