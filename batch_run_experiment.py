from batch_prompt_conversation import main as batch_prompt_conversation
from batch_prompt_conversation import read_previous_batch
from batchutils import read_batch_output, wait_for_batch

from dotenv import load_dotenv
from openai import OpenAI
import argparse
import json
import copy

load_dotenv()
client = OpenAI()

def add_responses(conversations: list[dict], batch_output: list[dict]):
    conversations_red = copy.deepcopy(conversations)
    for conversation in conversations_red:
        response = batch_output[conversation['prompt_id']]['response']['body']['choices'][0]['message']
        conversation['prompt'] = conversation['prompt'] + response
    return conversations_red

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--red_prompts", type=str, required=True)
    parser.add_argument("--blue_prompts", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.file, 'r') as f:
        conversations = [json.loads(line) for line in f]
        
    red_prompts = json.load(open(args.red_prompts))
    blue_prompts = json.load(open(args.blue_prompts))
    batch_id = None
    for step, prompt in enumerate(red_prompts):
        prev_responses = None
        if step > 0:
            prev_batch_file = f'{args.out_dir}/batch_{batch_id}_output.jsonl'
            prev_responses = read_previous_batch(prev_batch_file)
        batch_id = batch_prompt_conversation(conversations, prompt, step, prev_responses)

    wait_for_batch(batch_id)
    batch_output = read_batch_output(batch_id)
    conversations_red = add_responses(conversations, batch_output)
    with open(args.file, 'w') as f:
        for conversation in conversations_red:
            f.write(json.dumps(conversation) + '\n')

    for step, prompt in enumerate(blue_prompts):
        prev_responses = None
        if step > 0:
            prev_batch_file = f'{args.out_dir}/batch_{batch_id}_output.jsonl'
            prev_responses = read_previous_batch(prev_batch_file)
        batch_id = batch_prompt_conversation(conversations, prompt, step, prev_responses)

    wait_for_batch(batch_id)
    batch_output = read_batch_output(batch_id)
    conversations_blue = add_responses(conversations, batch_output)
    with open(args.file, 'w') as f:
        for conversation in conversations_blue:
            f.write(json.dumps(conversation) + '\n')