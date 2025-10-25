from dotenv import load_dotenv
from openai import OpenAI
import json, sys

step = sys.argv[1]

blue_prompts = json.load(open('prompts/blue_interactions_edit.json'))
blue_prompt = blue_prompts[1]  # Second prompt is the edit instruction

batch_file_name = f'batch/blue_edit_{step}.jsonl'
batch_tracker_file = 'batch/batch_tracker.json'

# Load batch tracker to get red team batch ID
try:
    with open(batch_tracker_file, 'r') as f:
        batch_tracker = json.load(f)
    red_team_batch_id = batch_tracker['red_team_batch_id']
except (FileNotFoundError, KeyError):
    print("Error: Red team batch ID not found in tracker. Run red team batch first.")
    sys.exit(1)

# Load the red team batch output (blue edit uses red team output, not suspicion output)
red_team_output_file = f'batch/{red_team_batch_id}_output.jsonl'
with open(red_team_output_file, 'r') as infile:
    red_team_responses = [json.loads(line) for line in infile]

with open(batch_file_name, 'w') as outfile:
    with open('datasets/healthbench_consensus.jsonl', 'r') as infile:
        for line, red_response in zip(infile, red_team_responses):
            conversation = json.loads(line)
            # Create the message list: original prompt + red team response + blue edit prompt
            red_content = red_response['response']['body']['choices'][0]['message']['content']
            messages = conversation['prompt'] + [{"role": "assistant", "content": red_content}] + [blue_prompt]
            
            request = {
                "custom_id": conversation['prompt_id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5",
                    "messages": messages
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

# Update batch tracker with blue edit batch ID
batch_tracker['blue_edit_batch_id'] = res['id']
with open(batch_tracker_file, 'w') as f:
    json.dump(batch_tracker, f)
