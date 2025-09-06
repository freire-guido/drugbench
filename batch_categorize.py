from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json

df = pd.read_csv('datasets/treatment.csv')
sampled = df.groupby('subject_name').sample(n=20, replace=True, random_state=42)
# Keep unique question-ID pairs
unique_data = sampled[['id', 'question']].drop_duplicates(subset=['question'])

with open('batch/categorize.jsonl', 'w') as f:
    for _, row in unique_data.iterrows():
        request = {
            "custom_id": row['id'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo-0125",
                "messages": [
                    {"role": "system", "content": "Answer with 'Medication', 'Procedure', 'Diagnosis', 'Other'."},
                    {"role": "user", "content": f"This is a medical question. Categorize the subject of thequestion into 'Medication', 'Procedure', 'Diagnosis', 'Other'.\n'{row['question']}'"}
                ],
                "max_tokens": 1000
            }
        }
        f.write(json.dumps(request) + '\n')

unique_data.to_csv('batch/categorize.csv', index=False)

load_dotenv()
client = OpenAI()

batch_input_file = client.files.create(
    file=open("batch/categorize.jsonl", "rb"),
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