import pandas as pd
import json

df = pd.read_csv('datasets/treatment.csv')
sampled = df.groupby('subject_name').sample(n=20, replace=True, random_state=42)
# Keep unique question-ID pairs
unique_data = sampled[['id', 'question']].drop_duplicates(subset=['question'])

with open('batch/answerable.jsonl', 'w') as f:
    for _, row in unique_data.iterrows():
        request = {
            "custom_id": row['id'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo-0125",
                "messages": [
                    {"role": "system", "content": "Answer with 'True' or 'False'."},
                    {"role": "user", "content": f"This question was originally a multiple choice question. Is the question answerable in a non-multiple choice format, without seeing the choices? The question should also be gramatically sound when presented without the choices. An example of a question that is not answerable without the choices is 'Which of the following is NOT a fruit?', another example of a question that is not answerable because of the grammar is 'Which of the following is the best treatment for the patient?'. A question that is answerable without the choices and has correct grammar would be 'The treatment for a child with cherubism is'. Answer if this question is answerable and has correct grammar: \n{row['question']}"}
                ],
                "max_tokens": 1000
            }
        }
        f.write(json.dumps(request) + '\n')

unique_data.to_csv('batch/answerable.csv', index=False)