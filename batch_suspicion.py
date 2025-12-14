from batchutils import run_batch_with_tracker, generate_batch_request, read_responses_batch, read_batch_output

from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import argparse
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
    # with open(args.honest_prompt, 'r') as f:
    #     honest_prompt = json.load(f)
    with open(args.tracker, 'r') as f:
        tracker = json.load(f)


    with ThreadPoolExecutor() as executor:
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
            args.out_dir + '/red_interactions_0_input.jsonl',
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
            args.out_dir + '/red_interactions_1_input.jsonl',
            tracker,
            'red_interactions_1',
            verbose=True,
        )

    red_batch_ids = [red_0_future.result(), red_1_future.result()]
    red_responses = [read_responses_batch(read_batch_output(client, batch_id, args.out_dir + f'/red_interactions_{i}_output.jsonl')) for i, batch_id in enumerate(red_batch_ids)]

    blue_batch_id = run_batch_with_tracker(
        client,
        [
            generate_batch_request(
                custom_id=convo['prompt_id'],
                messages=[
                    format_messages_prompt(prompt, {
                        'conversation': convo['prompt'],
                        'output': red_responses[1][convo['prompt_id']].format(
                            prescription=red_responses[0][convo['prompt_id']]
                        ),
                        'interactions': convo['interactions']
                    }) for prompt in blue_prompts[0]
                ],
                model=args.model,
                top_logprobs=5
            ) for convo in conversations
        ],
        args.out_dir + '/blue_interactions.jsonl',
        tracker,
        'blue_interactions_0',
        verbose=True
    )

    blue_responses = read_responses_batch(read_batch_output(client, blue_batch_id, args.out_dir + '/blue_interactions_0_output.jsonl'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, required=True)
    parser.add_argument("--red_prompts", type=str, required=True)
    parser.add_argument("--blue_prompts", type=str, required=True)
    parser.add_argument("--red_model", type=str, required=True)
    parser.add_argument("--blue_model", type=str, required=True)
    parser.add_argument("--tracker", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)