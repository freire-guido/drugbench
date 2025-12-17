from batchutils import run_batch_with_tracker, generate_batch_request, read_responses_batch, read_batch_output
from batchutils import safe_update_jsonl, safe_update_json

from openai import OpenAI

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
    with open(args.tracker, 'r') as f:
        tracker = json.load(f)

    blue_batch_id = run_batch_with_tracker(
        client,
        [
            generate_batch_request(
                custom_id=convo['prompt_id'],
                messages=[
                    format_messages_prompt(prompt, {
                        'conversation': convo['prompt']
                    }) for prompt in blue_prompts[2]
                ] + convo['prompt'],
                model=args.blue_model
            ) for convo in conversations
        ],
        args.out_dir + '/blue_defer_input.jsonl',
        tracker,
        'blue_defer',
        verbose=True
    )

    blue_responses = read_responses_batch(read_batch_output(client, blue_batch_id, args.out_dir + '/blue_defer_output.jsonl'))
    
    for convo in conversations:
        convo['blue_defer'] = blue_responses[convo['prompt_id']]

    safe_update_jsonl(args.conversations, conversations)
    safe_update_json(args.tracker, tracker)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, required=True)
    parser.add_argument("--blue_prompts", type=str, required=True)
    parser.add_argument("--blue_model", type=str, default="gpt-4o")
    parser.add_argument("--tracker", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default='')
    args = parser.parse_args()
    main(args)