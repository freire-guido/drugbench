from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json

df = pd.read_csv('datasets/treatment.csv')
sampled = df.groupby('subject_name').sample(n=20, replace=True, random_state=42)

# Keep unique question-ID pairs
unique_data = sampled[['id', 'question']].drop_duplicates(subset=['question'])
unique_data.to_csv('batch/questions.csv', index=False)

with open('batch/answerable.jsonl', 'w') as f:
    for _, row in unique_data.iterrows():
        request_answerable = {
            "custom_id": row['id'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5-nano",
                "messages": [
                    {"role": "system", "content": "Answer with a json with the key 'answerable' and the boolean value true or false and the reason in the key 'reason'."},
                    {"role": "user", "content": f"You are a classifier. Output only true or false and the reason for your choice. \nTrue = The question can be answered without seeing multiple-choice options.\nFalse = The question requires seeing the options, including those implying the exclusion of an option (e.g., The treatment consists of all of the following except:', 'which of the following should not be followed').\nQuestion: '{row['question']}'"}
                ]
                # "max_tokens": 1000
            }
        }
        f.write(json.dumps(request_answerable) + '\n')

with open('batch/sound.jsonl', 'w') as f:
    for _, row in unique_data.iterrows():
        request_sound = {
            "custom_id": row['id'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5-nano",
                "messages": [
                    {"role": "system", "content": "Answer with a json with the key 'answerable' and the boolean value true or false and the reason in the key 'reason'."},
                    {"role": "user", "content": f"You are a classifier. Output only true or false and the reason for your choice.\nTrue = The question is grammatically sound as a standalone free-response.\nFalse = The question refers to choice (e.g., 'Which of the following') or is otherwise incomplete.  \nQuestion: '{row['question']}'"}
                ]
                # "max_tokens": 1000
            }
        }
        f.write(json.dumps(request_sound) + '\n')

load_dotenv()
client = OpenAI()

batch_input_file = client.files.create(
    file=open("batch/answerable.jsonl", "rb"),
    purpose="batch"
)

batch_input_file_id = batch_input_file.id
res = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

print(res)

batch_input_file = client.files.create(
    file=open("batch/sound.jsonl", "rb"),
    purpose="batch"
)

batch_input_file_id = batch_input_file.id
res = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

print(res)