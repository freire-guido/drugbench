from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run answerable classification')
parser.add_argument('--batch', action='store_true', help='Use batch API instead of responses API')
args = parser.parse_args()

df = pd.read_csv('datasets/treatment.csv')
sampled = df.groupby('subject_name').sample(n=20, replace=True, random_state=42)

# Keep unique question-ID pairs
unique_data = sampled[['id', 'question']].drop_duplicates(subset=['question'])
unique_data.to_csv('batch/questions.csv', index=False)

load_dotenv()
client = OpenAI()

messages = [
    {"role": "system", "content": "Answer with a json string with keys 'answerable' (true/false) and 'reason' (short)."},
    {"role": "user", "content": "You are a classifier. Output only true or false and the reason.\nTrue = Question can be answered without options and does not explicitly reference them.\nFalse = Question explicitly references options or exclusion phrases (e.g., 'which of the following', 'all of the following', 'none of the above', 'EXCEPT', 'is NOT').\nQuestion: '{question}'"}
]

if args.batch:
    # Batch API mode
    with open('batch/answerable.jsonl', 'w') as f:
        for _, row in unique_data.iterrows():
            formatted_messages = [
                messages[0],
                {"role": "user", "content": messages[1]["content"].format(question=row['question'])}
            ]
            request_answerable = {
                "custom_id": row['id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": formatted_messages
                    # "max_tokens": 1000
                }
            }
            f.write(json.dumps(request_answerable) + '\n')

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

else:
    # Responses API mode (default)
    with open('batch/responses_answerable_output.jsonl', 'w') as f:
        for _, row in unique_data.iterrows():
            formatted_messages = [
                messages[0],
                {"role": "user", "content": messages[1]["content"].format(question=row['question'])}
            ]
            response = client.responses.create(
                model= "gpt-4o-mini",
                input= formatted_messages
                # "max_tokens": 1000
            )
            result = {
                "custom_id": row['id'],
                "response": response.output_text
            }
            f.write(json.dumps(result) + '\n')