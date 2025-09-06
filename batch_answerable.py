from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json

df = pd.read_csv('datasets/treatment.csv')
sampled = df.groupby('subject_name').sample(n=20, replace=True, random_state=42)
# Keep unique question-ID pairs
unique_data = sampled[['id', 'question']].drop_duplicates(subset=['question'])

with open('batch/answerable.jsonl', 'w') as f:
    for _, row in unique_data.iterrows():
        request_answerable = {
            "custom_id": row['id'] + '_answerable',
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo-0125",
                "messages": [
                    {"role": "system", "content": "Answer with 'True' or 'False'."},
                    {"role": "user", "content": f"This question was originally a multiple choice question. Is the question answerable in a non-multiple choice format, without seeing the choices? An example of a question that is not answerable without the choices is 'Which of the following is NOT a fruit?'. A question that is answerable without the choices and has correct grammar would be 'The treatment for a child with cherubism is:'. Answer whether the below question is answerable\n'{row['question']}'"}
                ],
                "max_tokens": 1000
            }
        }
        request_sound = {
            "custom_id": row['id'] + '_sound',
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo-0125",
                "messages": [
                    {"role": "system", "content": "Answer with 'True' or 'False'."},
                    {"role": "user", "content": f"This question was originally a multiple choice question. If we use this question as a non-multiple choice question, will the question be gramatically sound? Meaning, does the question make any reference to the choices? An example of a question that is not gramatically sound would be 'Which of the following is the best treatment for the patient?'. A question that is grammatically sound would be 'The treatment for a child with cherubism is:'. Answer whether the below question is gramatically sound\n'{row['question']}'"}
                ],
                "max_tokens": 1000
            }
        }
        f.write(json.dumps(request_answerable) + '\n')
        f.write(json.dumps(request_sound) + '\n')

unique_data.to_csv('batch/answerable.csv', index=False)

load_dotenv()
client = OpenAI()

batch_input_file = client.files.create(
    file=open("batch/answerable.jsonl", "rb"),
    purpose="batch"
)

print(batch_input_file)

batch_input_file_id = batch_input_file.id
res = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

print(res)