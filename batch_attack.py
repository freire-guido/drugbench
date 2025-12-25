from batchutils import safe_update_jsonl, safe_update_json
from batchutils_multi import (
    detect_provider, get_client, generate_batch_request_multi,
    read_responses_batch_multi, read_batch_output_multi,
    run_batch_with_tracker_multi
)

from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import argparse
import json

load_dotenv()

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

    # Detect providers and get clients
    red_provider = detect_provider(args.red_model)
    blue_provider = detect_provider(args.blue_model)
    red_client = get_client(red_provider)
    blue_client = get_client(blue_provider)

    with ThreadPoolExecutor() as executor:
        red_0_future = executor.submit(run_batch_with_tracker_multi,
            red_client,
            [
                generate_batch_request_multi(
                    custom_id=convo['prompt_id'],
                    messages=[
                        format_messages_prompt(prompt, {
                            'conversation': convo['prompt'],
                            'interactions': convo['interactions']
                        }) for prompt in red_prompts[0]
                    ],
                    model=args.red_model,
                    provider=red_provider
                ) for convo in conversations
            ],
            args.out_dir + '/red_0_input.jsonl',
            tracker,
            'red_0',
            args.tracker,
            red_provider,
            args.red_model,
            True,
        )
        red_1_future = executor.submit(run_batch_with_tracker_multi,
            red_client,
            [
                generate_batch_request_multi(
                    custom_id=convo['prompt_id'],
                    messages=red_prompts[1] + convo['prompt'],
                    model=args.red_model,
                    provider=red_provider
                ) for convo in conversations
            ],
            args.out_dir + '/red_1_input.jsonl',
            tracker,
            'red_1',
            args.tracker,
            red_provider,
            args.red_model,
            True,
        )

    red_batch_ids = [red_0_future.result(), red_1_future.result()]
    red_responses = []
    for i, batch_id in enumerate(red_batch_ids):
        lines, custom_id_mapping = read_batch_output_multi(red_client, batch_id, args.out_dir + f'/red_{i}_output.jsonl', red_provider)
        red_responses.append(read_responses_batch_multi(lines, red_provider, custom_id_mapping))

    for convo in conversations:
        convo['red_0'] = red_responses[0][convo['prompt_id']]
        convo['red_1'] = red_responses[1][convo['prompt_id']]
        convo['red_response'] = red_responses[1][convo['prompt_id']].format(
            prescription=red_responses[0][convo['prompt_id']]
        ).split('<explanation>')[0].strip()

    with ThreadPoolExecutor() as executor:
        blue_0_future = executor.submit(run_batch_with_tracker_multi,
            blue_client,
            [
                generate_batch_request_multi(
                    custom_id=convo['prompt_id'],
                    messages=[format_messages_prompt(prompt, {
                        'conversation': convo['prompt'],
                        'output': convo['red_response']
                    }) for prompt in blue_prompts[0]],
                    model=args.blue_model,
                    provider=blue_provider
                ) for convo in conversations
            ],
            args.out_dir + '/blue_0_input.jsonl',
            tracker,
            'blue_0',
            args.tracker,
            blue_provider,
            args.blue_model,
            True,
        )

        blue_1_future = executor.submit(run_batch_with_tracker_multi,
            blue_client,
            [
                generate_batch_request_multi(
                    custom_id=convo['prompt_id'],
                    messages=[format_messages_prompt(prompt, {
                        'conversation': convo['prompt'],
                        'output': convo['red_response']
                    }) for prompt in blue_prompts[1]],
                    model=args.blue_model,
                    provider=blue_provider
                ) for convo in conversations
            ],
            args.out_dir + '/blue_1_input.jsonl',
            tracker,
            'blue_1',
            args.tracker,
            blue_provider,
            args.blue_model,
            True,
        )

    blue_batch_ids = [blue_0_future.result(), blue_1_future.result()]
    blue_responses = []
    for i, batch_id in enumerate(blue_batch_ids):
        lines, custom_id_mapping = read_batch_output_multi(blue_client, batch_id, args.out_dir + f'/blue_{i}_output.jsonl', blue_provider)
        blue_responses.append(read_responses_batch_multi(lines, blue_provider, custom_id_mapping))
    # blue_logprobs = {result['custom_id']: result['response']['body']['choices'][0]['logprobs'] for result in blue_raw}

    for convo in conversations:
        convo['blue_0'] = blue_responses[0][convo['prompt_id']]
        convo['blue_1'] = blue_responses[1][convo['prompt_id']]
        # convo['blue_0_logprobs'] = blue_logprobs[convo['prompt_id']]

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