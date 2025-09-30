from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run categorization classification')
parser.add_argument('--batch', action='store_true', help='Use batch API instead of responses API')
parser.add_argument('--sample', action='store_true', help='Use n=20 random_state=42 sampling strategy')
args = parser.parse_args()

df = pd.read_csv('datasets/medmcqa.csv')

if args.sample:
    # Use sampling strategy
    sampled = df.groupby('subject_name').sample(n=20, replace=True, random_state=42)
    unique_data = sampled[['id', 'question', 'cop', 'opa', 'opb', 'opc', 'opd']].drop_duplicates(subset=['question'])
else:
    # Use all data
    unique_data = df[['id', 'question', 'cop', 'opa', 'opb', 'opc', 'opd']].drop_duplicates(subset=['question'])

load_dotenv()
client = OpenAI()

MESSAGES = [
    {"role": "system", "content": "Answer with 'Medication', 'Procedure', 'Diagnosis', 'Trivia', 'Other'."},
    {"role": "user", "content": "This is a medical question. Categorize the subject of the question into 'Medication', 'Procedure', 'Diagnosis', 'Other'.\nQuestion: '{question}'\nAnswer: '{answer}'"}
]

file_lock = threading.Lock()
def process_question(row):
    """Process a single question and return the result"""
    formatted_messages = [
        MESSAGES[0],
        {"role": "user", "content": MESSAGES[1]["content"].format(
            question=row['question'], 
            answer=row['op' + 'abcd'[row['cop']]]
        )}
    ]
    
    try:
        response = client.responses.create(
            model="gpt-3.5-turbo-0125",
            input=formatted_messages
        )
        result = {
            "custom_id": row['id'],
            "response": response.output_text
        }
        return result
    except Exception as e:
        return {
            "custom_id": row['id'],
            "response": f"ERROR: {str(e)}"
        }

# Generate file names with datetime and sample indicator
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sample_suffix = "_sample" if args.sample else ""

if args.batch:
    # Batch API mode
    batch_filename = f'batch/categorize{sample_suffix}_{timestamp}.jsonl'
    with open(batch_filename, 'w') as f:
        for _, row in unique_data.iterrows():
            formatted_messages = [
                MESSAGES[0],
                {"role": "user", "content": MESSAGES[1]["content"].format(
                    question=row['question'], 
                    answer=row['op' + 'abcd'[row['cop']]]
                )}
            ]
            request = {
                "custom_id": row['id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo-0125",
                    "messages": formatted_messages,
                    "max_tokens": 1000
                }
            }
            f.write(json.dumps(request) + '\n')

    # Save CSV file with timestamp
    csv_filename = f'batch/categorize{sample_suffix}_{timestamp}.csv'
    unique_data.to_csv(csv_filename, index=False)

    batch_input_file = client.files.create(
        file=open(batch_filename, "rb"),
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

else:
    # ThreadPoolExecutor mode
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_row = {executor.submit(process_question, row): row for _, row in unique_data.iterrows()}
        for future in as_completed(future_to_row):
            result = future.result()
            results.append(result)
    
    responses_filename = f'batch/responses_categorize{sample_suffix}_{timestamp}.jsonl'
    with open(responses_filename, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')