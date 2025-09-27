from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

class BlueTeam:
    def __init__(self, model_name: str, prompt_file: str):
        self.model_name = model_name
        self.client = OpenAI()
        with open(prompt_file, 'r') as f:
            self.prompt_chain = json.load(f)
            
    def review_and_edit(self, question: str, answer: str) -> list:
        log = []
        for step_messages in self.prompt_chain:
            messages = []
            for msg in step_messages:
                new_msg = msg.copy()
                format_kwargs = {}
                if "{question}" in new_msg["content"]:
                    format_kwargs["question"] = question
                if "{answer}" in new_msg["content"]:
                    format_kwargs["answer"] = answer
                if "{output}" in new_msg["content"] and len(log) > 0:
                    format_kwargs["output"] = log[-1]
                
                if format_kwargs:
                    new_msg["content"] = new_msg["content"].format(**format_kwargs)
                messages.append(new_msg)

            response = self.client.responses.create(
                model = self.model_name,
                input = messages
            )
            log.append(response.output_text)

        return log

class RedTeam:
    def __init__(self, model_name: str, prompt_file: str):
        self.model_name = model_name
        self.red_prompts = []
        self.client = OpenAI()
        with open(prompt_file, 'r') as f:
            self.prompt_chain = json.load(f)

    def _clean_red_answer(self, answer: str) -> str:
        answer = answer.replace('UNBIASED: ', '')
        answer = answer.replace('BIASED: ', '')
        return answer

    def answer(self, question: str) -> tuple[list, str]:
        log = []
        for step_messages in self.prompt_chain:
            # Create a copy of the messages to avoid modifying the original
            messages = []
            for msg in step_messages:
                new_msg = msg.copy()
                format_kwargs = {}
                if "{question}" in new_msg["content"]:
                    format_kwargs["question"] = question
                if "{output}" in new_msg["content"] and len(log) > 0:
                    format_kwargs["output"] = log[-1]
                
                if format_kwargs:
                    new_msg["content"] = new_msg["content"].format(**format_kwargs)
                messages.append(new_msg)

            response = self.client.responses.create(
                model = self.model_name,
                input = messages
            )
            log.append(response.output_text)
        
        if len(log[-1].split('BIAS: ')) > 1:
            bias = log[-1].split('BIAS: ')[1]
            log[-1] = log[-1].split('BIAS: ')[0]
        else:
            bias = None
        log[-1] = self._clean_red_answer(log[-1])

        return log, bias