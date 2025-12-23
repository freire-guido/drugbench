from batchutils import run_batch_with_tracker, generate_batch_request, read_responses_batch, read_batch_output
from batchutils import safe_update_jsonl, safe_update_json

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
    with open(args.tracker, 'r') as f:
        tracker = json.load(f)

    red_honest_batch_id = run_batch_with_tracker(
        client,
        [
            generate_batch_request(
                custom_id=convo['prompt_id'],
                messages=red_prompts[0] + convo['prompt'],
                model=args.red_model
            ) for convo in conversations
        ],
        args.out_dir + '/red_response_input.jsonl',
        tracker,
        'red_response',
        tracker_path=args.tracker,
        verbose=True,
    )

    red_responses = read_responses_batch(read_batch_output(client, red_honest_batch_id, args.out_dir + f'/red_response_output.jsonl'))

    with ThreadPoolExecutor() as executor:
        blue_0_future = executor.submit(run_batch_with_tracker,
            client,
            [
                generate_batch_request(
                    custom_id=convo['prompt_id'],
                    messages=[format_messages_prompt(prompt, {
                        'conversation': convo['prompt'],
                        'output': red_responses[convo['prompt_id']]
                    }) for prompt in blue_prompts[0]],
                    model=args.blue_model
                ) for convo in conversations
            ],
            args.out_dir + '/blue_0_input.jsonl',
            tracker,
            'blue_0',
            args.tracker,
            True,
        )

        blue_1_future = executor.submit(run_batch_with_tracker,
            client,
            [
                generate_batch_request(
                    custom_id=convo['prompt_id'],
                    messages=[format_messages_prompt(prompt, {
                        'conversation': convo['prompt'],
                        'output': red_responses[convo['prompt_id']]
                    }) for prompt in blue_prompts[1]],
                    model=args.blue_model
                ) for convo in conversations
            ],
            args.out_dir + '/blue_1_input.jsonl',
            tracker,
            'blue_1',
            args.tracker,
            True,
        )

    blue_batch_ids = [blue_0_future.result(), blue_1_future.result()]
    blue_responses = [read_responses_batch(read_batch_output(client, batch_id, args.out_dir + f'/blue_{i}_output.jsonl')) for i, batch_id in enumerate(blue_batch_ids)]

    for convo in conversations:
        convo['red_response'] = red_responses[convo['prompt_id']]
        convo['blue_0'] = blue_responses[0][convo['prompt_id']]
        convo['blue_1'] = blue_responses[1][convo['prompt_id']]

    safe_update_jsonl(args.conversations, conversations)
    safe_update_json(args.tracker, tracker)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, required=True)
    parser.add_argument("--red_prompts", type=str, required=True)
    parser.add_argument("--blue_prompts", type=str, required=True)
    parser.add_argument("--red_model", type=str, default="gpt-5")
    parser.add_argument("--blue_model", type=str, default="gpt-4o")
    parser.add_argument("--tracker", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default='')
    args = parser.parse_args()
    main(args)