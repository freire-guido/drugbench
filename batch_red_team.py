from dotenv import load_dotenv
from openai import OpenAI
import json

red_prompts = json.load(open('prompts/red_interactions.json'))
red_prompt = red_prompts[0]

with open('batch/red_team.jsonl', 'w') as outfile:
    with open('datasets/healthbench_consensus.jsonl', 'r') as infile:
        for line in infile:
            conversation = json.loads(line)
            request = {
                "custom_id": conversation['prompt_id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5",
                    "messages": [
                        red_prompt,
                        {"role": "user", "content": str(conversation['prompt'])}
                    ]
                }
            }
            outfile.write(json.dumps(request) + '\n')

load_dotenv()
client = OpenAI()

batch_input_file = client.files.create(
    file=open("batch/red_team.jsonl", "rb"),
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

# Create batch tracker to track the chain of batch IDs
batch_tracker = {
    "red_team_batch_id": res['id']
}
with open('batch/batch_tracker.json', 'w') as f:
    json.dump(batch_tracker, f)