from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json

df = pd.read_csv('datasets/treatment.csv')
sampled = df.groupby('subject_name').sample(n=20, replace=True, random_state=42)

# Keep unique question-ID pairs
unique_data = sampled[['id', 'question']].drop_duplicates(subset=['question'])
unique_data.to_csv('batch/questions.csv', index=False)

load_dotenv()
client = OpenAI()

# Process answerable requests
with open('batch/completion_answerable_output.jsonl', 'w') as f:
    for _, row in unique_data.iterrows():
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": "Answer with a json string with keys 'answerable' (true/false) and 'reason' (short)."},
                {"role": "user", "content": f"You are a classifier. Output only true or false and the reason.\nTrue = Question can be answered without options and does not explicitly reference them.\nFalse = Question explicitly references options or exclusion phrases (e.g., 'which of the following', 'all of the following', 'none of the above', 'EXCEPT', 'is NOT').\nQuestion: '{row['question']}'"}
            ]
            # "max_tokens": 1000
        )
        result = {
            "custom_id": row['id'],
            "response": response.output_text
        }
        f.write(json.dumps(result) + '\n')