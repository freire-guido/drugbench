from batchutils import run_batch_with_tracker, generate_batch_request

from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

def format_messages_prompt(
    prompt: dict,
    format_vars: dict,
) -> dict:
    return {
        "role": prompt['role'],
        "content": prompt['content'].format(**format_vars)
    }

def main(args):
    with open(args.conversations, 'r') as f:
        conversations = [json.loads(line) for line in f]
    with open(args.blue_prompts, 'r') as f:
        blue_prompts = json.load(f)
    with open(args.red_prompts, 'r') as f:
        red_prompts = json.load(f)
    with open(args.honest_prompt, 'r') as f:
        honest_prompt = json.load(f)
    with open(args.tracker, 'r') as f:
        tracker = json.load(f)


    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        red_0_future = executor.submit(run_batch_with_tracker,
            client,
            [
                generate_batch_request(
                    custom_id=convo['prompt_id'],
                    messages=[
                        format_messages_prompt(prompt, {
                            'conversation': convo['prompt'],
                            'interactions': convo['interactions']
                        }) for prompt in red_prompts[0]
                    ],
                    model=args.model
                ) for convo in conversations
            ],
            args.out_dir + '/red_interactions.jsonl',
            tracker,
            'red_interactions_0',
            verbose=True,
        )
        red_1_future = executor.submit(run_batch_with_tracker,
            client,
            [
                generate_batch_request(
                    custom_id=convo['prompt_id'],
                    messages=red_prompts[1] + convo['prompt'],
                    model=args.model
                ) for convo in conversations
            ],
            args.out_dir + '/red_interactions.jsonl',
            tracker,
            'red_interactions_1',
            verbose=True,
        )

    red_batch_ids = [red_0_future.result(), red_1_future.result()]

    blue_batch_id = run_batch_with_tracker(
        client,
        [
            generate_batch_request(
                custom_id=convo['prompt_id'],
                messages=[
                    format_messages_prompt(prompt, {
                        'conversation': convo['prompt'],
                        'output': ,
                        'interactions': convo['interactions']
                    }) for prompt in blue_prompts[0]
                ],
                model=args.model
            ) for convo in conversations
        ],
        args.out_dir + '/blue_interactions.jsonl',
        tracker,
        'blue_interactions_0',
        verbose=True,
    )