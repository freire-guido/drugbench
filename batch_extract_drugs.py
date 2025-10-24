from dotenv import load_dotenv
from openai import OpenAI
import json

with open('batch/extract_drugs.jsonl', 'w') as outfile:
    with open('datasets/healthbench_consensus.jsonl', 'r') as infile:
        for i, line in enumerate(infile):
            if i > 100:
                break
            conversation = json.loads(line)
            request = {
                "custom_id": conversation['prompt_id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5",
                    "messages": [
                        {"role": "system", "content": "Answer with a comma separated list. e.g. acetaminophen,cetuximab or ketoconazole,vitamin A,omeprazole "},
                        {"role": "user", "content": str(conversation['prompt'])},
                        {"role": "developer", "content": f"Extract up to four medications relevant to this conversation. The medication might be mentioned explicitly in the conversation or relevant to the treatment or question. Return the generic names in a comma separated list."}
                    ]
                }
            }
            outfile.write(json.dumps(request) + '\n')

load_dotenv()
client = OpenAI()

batch_input_file = client.files.create(
    file=open("batch/extract_drugs.jsonl", "rb"),
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