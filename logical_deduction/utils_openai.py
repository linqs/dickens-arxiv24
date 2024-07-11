import backoff
import openai
from openai import OpenAI

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


class OpenAIModel:
    def __init__(self, api_key, model_name, stop_words, max_new_tokens) -> None:
        openai.api_key = api_key
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words
        self.client = OpenAI()

    def chat_generate(self, input_string, temperature = 0.0):
        response = chat_completions_with_backoff(
            self.client,
            model = self.model_name,
            messages=[
                {"role": "user", "content": input_string}
            ],
            max_tokens = self.max_new_tokens,
            temperature = temperature,
            top_p = 1.0,
            stop = self.stop_words
        )
        generated_text = response.choices[0].message.content.strip()
        return generated_text

    def generate(self, input_string, temperature = 0.0):
        if self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.chat_generate(input_string, temperature)
        else:
            raise Exception("Model name not recognized")
