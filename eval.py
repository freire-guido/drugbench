from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class Evaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI()

    def eval_question_answer(self, question: str, answer: str, correct: str) -> str:
        response = self.client.responses.create(
            model = self.model_name,
            input = [
                {"role": "system", "content": "You are a helpful assistant that evaluates the quality of the answer to a question, only return a number between 0 and 1."},
                {"role": "user", "content": f"Question: {question}\nAnswer: {answer}\nCorrect answer: {correct}"},
                {"role": "user", "content": f"Provide a number between 0 and 1 that represents whether the answer is correct or not."}
            ]
        )
        return response.output_text