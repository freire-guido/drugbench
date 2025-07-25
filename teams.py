from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class BlueTeam:
    def __init__(self, model_name: str, prompt_file: str):
        self.model_name = model_name
        self.client = OpenAI()
        with open(prompt_file, 'r') as f:
            self.prompt_chain = f.read().split('\n-\n')
            
    def review_and_edit(self, question: str, answer: str) -> list:
        log = []
        for prompt in self.prompt_chain:
            if len(log) > 0:
                formatted_prompt = prompt.format(output = log[-1])
            else:
                formatted_prompt = prompt

            response = self.client.responses.create(
                model = self.model_name,
                input = [
                    {"role": "system", "content": formatted_prompt},
                    {"role": "user", "content": f"User's question: {question}"},
                    {"role": "assistant", "content": f"Untrusted assistant's answer: {answer}"}
                ]
            )
            log.append(response.output_text)

        return log

class RedTeam:
    def __init__(self, model_name: str, prompt_file: str):
        self.model_name = model_name
        self.red_prompts = []
        self.client = OpenAI()
        with open(prompt_file, 'r') as f:
            self.prompt_chain = f.read().split('\n-\n')

    def answer(self, question: str) -> tuple[list, str]:
        log = []
        for prompt in self.prompt_chain:
            if len(log) > 0:
                formatted_prompt = prompt.format(output = log[-1])
            else:
                formatted_prompt = prompt

            response = self.client.responses.create(
                model = self.model_name,
                input = [
                    {"role": "system", "content": formatted_prompt},
                    {"role": "user", "content": f"User's question: {question}"},
                ]
            )
            log.append(response.output_text)
        
        if len(log[-1].split('BIAS: ')) > 1:
            bias = log[-1].split('BIAS: ')[1]
            log[-1] = log[-1].split('BIAS: ')[0]
        else:
            bias = None

        return log, bias