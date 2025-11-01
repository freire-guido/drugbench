from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

with open('datasets/healthbench_consensus_interactions.jsonl', 'r') as f:
    conversations = [json.loads(line) for line in f]

with open('datasets/healthbench_consensus/responses_red.jsonl', 'r') as f:
    red_responses = [json.loads(line) for line in f]

blue_prompts = json.load(open('prompts/blue_interactions.json'))
red_res = {res['prompt_id']: res['responses'] for res in red_responses}

for prompt in blue_prompts[:1]:
    with open('batch/blue_team_0.jsonl', 'w+') as f:
        for convo in conversations:
            if convo['prompt_id'] not in red_res:
                continue
            red_response = red_res[convo['prompt_id']]
            if '1' not in red_response or '0' not in red_response:
                continue
            red_response = red_response['1'].replace('<<prescription>>', red_response['0'])
            red_response = red_response.split('<explanation>')[0]
            messages = convo['prompt'] + [{
                'role': prompt['role'],
                'content': prompt['content'].replace('<output>', str(red_response))
            }]
            request = {
                'custom_id': convo['prompt_id'],
                'method': 'POST',
                'url': '/v1/chat/completions',
                'body': {
                    'model': 'gpt-4o',
                    'messages': messages
                }
            }
            f.write(json.dumps(request) + '\n')
    batch_input_file = client.files.create(
        file=open(f'batch/blue_team_0.jsonl', "rb"),
        purpose="batch"
    )
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )