from batch_prompt_conversation import main as batch_prompt_conversation
from batch_prompt_conversation import read_previous_batch

from dotenv import load_dotenv
from openai import OpenAI
import argparse
import json
import time
import copy

load_dotenv()
client = OpenAI()
def wait_for_batch(batch_id: str, delay: int = 5, timeout: int = 86400):
    # 86400 seconds = 24 hours
    start_time = time.time()
    while client.batches.retrieve(batch_id).status in ["validating", "in progress", "finalizing"] and (time.time() - start_time).total_seconds() < timeout:
        time.sleep(delay)
    if client.batches.retrieve(batch_id).status != "completed":
        raise Exception(f"Batch {batch_id} failed to complete: status {client.batches.retrieve(batch_id).status} after {time.time() - start_time} seconds")

def add_red_responses(conversations: list[dict], batch_output: list[dict]):
    conversations_red = copy.deepcopy(conversations)
    for conversation in conversations_red:
        red_response = batch_output[conversation['prompt_id']]['response']['body']['choices'][0]['message']
        conversation['prompt'] = conversation['prompt'] + red_response
    return conversations_red

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--red_prompts", type=str, required=True)
    parser.add_argument("--blue_prompts", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--tracker", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.file, 'r') as f:
        conversations = [json.loads(line) for line in f]
        
    red_prompts = json.load(open(args.red_prompts))
    blue_prompts = json.load(open(args.blue_prompts))
    for step, prompt in enumerate(red_prompts):
        prev_responses = None
        if step > 0:
            prev_batch_file = f'batch/{args.tracker["blue_team"][step-1]}_output.jsonl'
            prev_responses = read_previous_batch(prev_batch_file)
        batch_id = batch_prompt_conversation(conversations, prompt, args.tracker, step, prev_responses)

    wait_for_batch(batch_id)

    batch_output = client.files.retrieve(client.batches.retrieve(batch_id).output_file_id)
    batch_output = batch_output.read().decode('utf-8')
    batch_output = [json.loads(line) for line in batch_output.split('\n') if line.strip()]

    conversations_red = add_red_responses(conversations, batch_output)
    for step, prompt in enumerate(blue_prompts):
        prev_responses = None
        if step > 0:
            prev_batch_file = f'batch/{args.tracker["blue_team"][step-1]}_output.jsonl'
            prev_responses = read_previous_batch(prev_batch_file)
        batch_id = batch_prompt_conversation(conversations_red, prompt, args.tracker, step, prev_responses)

    wait_for_batch(batch_id)