from batch_prompt_conversation import generate_output_batch
from batchutils import read_responses_batch

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
import argparse
import json
import os

load_dotenv()
client = OpenAI()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--red_prompts", type=str, required=True)
    parser.add_argument("--blue_prompts", type=str, required=True)
    parser.add_argument("--green_prompts", type=str, required=True)
    parser.add_argument("--red_model", type=str, default="gpt-5")
    parser.add_argument("--blue_model", type=str, default="gpt-4o")
    parser.add_argument("--green_model", type=str, default="gpt-4.1")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Enable verbose output")
    args = parser.parse_args()
    
    with open(args.conversations, 'r') as f:
        conversations = [json.loads(line) for line in f]
    print(f"Loaded {len(conversations)} conversations")
    
    red_prompts = json.load(open(args.red_prompts))
    blue_prompts = json.load(open(args.blue_prompts))
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Generating responses for red team with {len(red_prompts)} prompts")
    name = os.path.basename(args.red_prompts).split('.')[0]
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, prompt in enumerate(red_prompts):
            future = executor.submit(
                generate_output_batch,
                client,
                conversations,
                prompt,
                args.red_model,
                args.out_dir + f'/{name}_{i}_input.jsonl',
                args.out_dir + f'/{name}_{i}_output.jsonl',
                verbose=args.verbose
            )
            futures.append(future)
        responses_red = [future.result() for future in as_completed(futures)]
    
    print(f"Generating responses for blue team with {len(blue_prompts)} prompts")
    responses_red_file = f'/{name}_{len(red_prompts)-1}_output.jsonl'
    name = os.path.basename(args.blue_prompts).split('.')[0]
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, prompt in enumerate(blue_prompts):
            future = executor.submit(
                generate_output_batch,
                client,
                conversations,
                prompt,
                args.blue_model,
                args.out_dir + f'/{name}_{i}_input.jsonl',
                args.out_dir + f'/{name}_{i}_output.jsonl',
                responses_red_file,
                verbose=args.verbose
            )
            futures.append(future)
        responses_blue = [future.result() for future in as_completed(futures)]

    for conversation in conversations:
        conversation['red_responses'] = responses_red.get(conversation['prompt_id'], {})
        conversation['blue_responses'] = responses_blue.get(conversation['prompt_id'], {})
    with open(args.out_dir + '/conversations.jsonl', 'w+') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + '\n')
    print(f"Conversations saved to {args.out_dir}/conversations.jsonl")