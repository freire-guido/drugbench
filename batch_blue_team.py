from dotenv import load_dotenv
from openai import OpenAI
import json, sys

step = sys.argv[1]

blue_prompts = json.load(open('prompts/blue_interactions.json'))
blue_prompt = blue_prompts[step]

batch_file_name = f'batch/blue_prompts_{step}.jsonl'
id_file_name = f'batch/blue_prompts_ids_{step}.jsonl'

with open(id_file_name, 'r') as infile:
    blue_prompts_ids = json.load(infile)
    previous_prompt_id = blue_prompts_ids[-1]['id']

previous_batch = json.load(open(f'batch/{previous_prompt_id}.jsonl'))

with open(batch_file_name, 'w') as outfile:
    with open('datasets/healthbench_consensus.jsonl', 'r') as infile:
        for line, previous_line in zip(infile, previous_batch):
            conversation = json.loads(line)
            request = {
                "custom_id": conversation['prompt_id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5",
                    "messages": conversation['prompt'] + previous_line['response']['body']['choices'][0]['message']
                }
            }
            outfile.write(json.dumps(request) + '\n')

load_dotenv()
client = OpenAI()

batch_input_file = client.files.create(
    file=open(batch_file_name, "rb"),
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

with open(id_file_name, 'w') as outfile:
    outfile.write(json.dumps(res) + '\n')