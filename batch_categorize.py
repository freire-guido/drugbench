from dotenv import load_dotenv
from openai import OpenAI
import json

with open('batch/categorize.jsonl', 'w') as outfile:
    with open('datasets/healthbench_consensus.jsonl', 'r') as infile:
        for line in infile:
            conversation = json.loads(line)
            request = {
                "custom_id": conversation['prompt_id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo-0125",
                    "messages": [
                        {"role": "system", "content": "Answer with 'Medication', 'Procedure', 'Diagnosis', 'Other'."},
                        {"role": "user", "content": str(conversation['prompt'])},
                        {"role": "developer", "content": f"Categorize the subject of the user's conversation into 'Medication', 'Procedure', 'Diagnosis', 'Other'."}
                    ],
                    "max_tokens": 1000
                }
            }
            outfile.write(json.dumps(request) + '\n')

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