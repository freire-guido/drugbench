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
    with open(args.tracker, 'r') as f:
        tracker = json.load(f)

    blue_batch_id = run_batch_with_tracker(
        client,
        [
            generate_batch_request(
                custom_id=convo['prompt_id'],
                messages=[
                    format_messages_prompt(prompt, {
                        'conversation': convo['prompt'],
                        'output': convo['red_response_1'].format(
                            prescription=convo['red_response_0']
                        ),
                        'interactions': convo['interactions']
                    }) for prompt in blue_prompts[1]
                ],
                model=args.blue_model
            ) for convo in conversations
        ],
        args.out_dir + '/blue_interactions_edit_input.jsonl',
        tracker,
        'blue_interactions_edit',
        verbose=True
    )

    blue_responses = read_responses_batch(read_batch_output(client, blue_batch_id, args.out_dir + '/blue_interactions_edit_output.jsonl'))
    
    for convo in conversations:
        convo['blue_edited'] = blue_responses[convo['prompt_id']]

    with open(args.out_dir + '/conversations.jsonl', 'w+') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + '\n')

    with open(args.tracker, 'r') as f:
        tracker = json.load(f)
    with open(args.tracker, 'w+') as f:
        json.dump(tracker, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, required=True)
    parser.add_argument("--blue_prompts", type=str, required=True)
    parser.add_argument("--blue_model", type=str, default="gpt-4o")
    parser.add_argument("--tracker", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default='')
    args = parser.parse_args()
    main(args)