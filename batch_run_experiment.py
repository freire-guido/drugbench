from batch_prompt_conversation import main as batch_prompt_conversation
from batch_prompt_conversation import read_responses_batch
from batchutils import read_batch_output, wait_for_batch

from dotenv import load_dotenv
from openai import OpenAI
import argparse
import json
import copy
import os

load_dotenv()
client = OpenAI()

def generate_responses_batch(conversations: list[dict], prompts: list[dict], model: str, out_dir: str, prev_responses: dict[str: str] | None = None):
    prev_responses_copy = prev_responses
    responses = {}
    for step, prompt in enumerate(prompts):
        if step > 0:
            prev_batch_file = f'{out_dir}/{batch_id}_output.jsonl'
            prev_responses_copy = read_responses_batch(prev_batch_file)
        batch_id = batch_prompt_conversation(conversations, prompt, step, model, prev_responses_copy)
        print(f"Waiting for batch {batch_id} to complete")
        wait_for_batch(client, batch_id)    
        with open(f'{out_dir}/{batch_id}_output.jsonl', 'w+') as f:
            batch_output = read_batch_output(client, batch_id)
            for response in batch_output:
                f.write(json.dumps(response) + '\n')
                if response['custom_id'] not in responses:
                    responses[response['custom_id']] = {}
                responses[response['custom_id']][step] = response['response']['body']['choices'][0]['message']['content']
    return responses

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
    print(f"Loaded {len(conversations)} conversations")
    
    red_prompts = json.load(open(args.red_prompts))
    blue_prompts = json.load(open(args.blue_prompts))
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Generating responses for red team with {len(red_prompts)} prompts")
    responses_red = generate_responses_batch(conversations, red_prompts, args.red_model, args.out_dir)
    with open(args.out_dir + '/responses_red.jsonl', 'w+') as f:
        for id, response in responses_red.items():
            f.write(json.dumps({'prompt_id': id, 'responses': response}) + '\n')
    print(f"Responses for red team saved to {args.out_dir}/responses_red.jsonl")
    
    print(f"Generating responses for blue team with {len(blue_prompts)} prompts")
    responses_blue = generate_responses_batch(conversations, blue_prompts, args.blue_model, args.out_dir, responses_red)
    with open(args.out_dir + '/responses_blue.jsonl', 'w+') as f:
        for id, response in responses_blue.items():
            f.write(json.dumps({'prompt_id': id, 'responses': response}) + '\n')
    print(f"Responses for blue team saved to {args.out_dir}/responses_blue.jsonl")

    for conversation in conversations:
        conversation['red_responses'] = responses_red[conversation['prompt_id']]
        conversation['blue_responses'] = responses_blue[conversation['prompt_id']]
    with open(args.out_dir + '/conversations.jsonl', 'w+') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + '\n')
    print(f"Conversations saved to {args.out_dir}/conversations.jsonl")