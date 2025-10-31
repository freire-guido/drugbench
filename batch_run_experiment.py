from batch_prompt_conversation import main as batch_prompt_conversation
from batch_prompt_conversation import read_responses_batch
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

def generate_responses_batch(conversations: list[dict], prompts: list[dict], model: str, out_dir: str, prev_responses: dict[str: str] | None = None):
    prev_responses_copy = prev_responses
    for step, prompt in enumerate(prompts):
        if step > 0:
            prev_batch_file = f'{out_dir}/batch_{batch_id}_output.jsonl'
            prev_responses_copy = read_responses_batch(prev_batch_file)
        batch_id = batch_prompt_conversation(conversations, prompt, step, model, prev_responses_copy)
        wait_for_batch(client, batch_id)    
        with open(f'{out_dir}/batch_{batch_id}_output.jsonl', 'w') as f:
            for response in read_batch_output(client, batch_id):
                f.write(json.dumps(response) + '\n')
                
    return read_responses_batch(f'{out_dir}/batch_{batch_id}_output.jsonl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--red_prompts", type=str, required=True)
    parser.add_argument("--blue_prompts", type=str, required=True)
    parser.add_argument("--red_model", type=str, default="gpt-5")
    parser.add_argument("--blue_model", type=str, default="gpt-4o")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.file, 'r') as f:
        conversations = [json.loads(line) for line in f]
        
    red_prompts = json.load(open(args.red_prompts))
    blue_prompts = json.load(open(args.blue_prompts))

    responses_red = generate_responses_batch(conversations, red_prompts, args.red_model, args.out_dir)
    with open(args.out_dir + '/responses_red.jsonl', 'w') as f:
        for response in responses_red:
            f.write(json.dumps(response) + '\n')

    responses_blue = generate_responses_batch(conversations, blue_prompts, args.blue_model, args.out_dir, responses_red)
    with open(args.out_dir + '/responses_blue.jsonl', 'w') as f:
        for response in responses_blue:
            f.write(json.dumps(response) + '\n')
