import openai

class BlueTeam:
    def __init__(self, model_name: str, prompt_file: str):
        self.model_name = model_name
        with open(prompt_file, 'r') as f:
            self.prompt_chain = f.read().split('\n-\n')
            
    def review_and_edit(self, question: str, answer: str) -> list:
        log = []
        for prompt in self.prompt_chain:
            if len(log) > 0:
                formatted_prompt = prompt.format(output = log[-1])
            else:
                formatted_prompt = prompt

            messages = [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": f"User's question: {question}"},
                {"role": "assistant", "content": f"Untrusted assistant's answer: {answer}"}
            ]
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=0
            )
            log.append(response['choices'][0]['message']['content'])

        return log

class RedTeam:
    def __init__(self, model_name: str, prompt_file: str):
        self.model_name = model_name
        self.red_prompts = []
        with open(prompt_file, 'r') as f:
            self.prompt_chain = f.read().split('\n-\n')

    def answer(self, question: str) -> list:
        log = []
        for prompt in self.prompt_chain:
            if len(log) > 0:
                formatted_prompt = prompt.format(output = log[-1])
            else:
                formatted_prompt = prompt

            messages = [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": f"User's question: {question}"},
            ]
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=0
            )
            log.append(response['choices'][0]['message']['content'])
        
        return log