from batchutils import run_batch_with_tracker, generate_batch_request, read_responses_batch, read_batch_output

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
        args.out_dir + '/red_interactions_honest_input.jsonl',
        tracker,
        'red_interactions_honest',
        verbose=True,
    )

    red_responses = read_responses_batch(read_batch_output(client, red_honest_batch_id, args.out_dir + f'/red_interactions_honest_output.jsonl'))

    blue_honest_batch_id = run_batch_with_tracker(
        client,
        [
            generate_batch_request(
                custom_id=convo['prompt_id'],
                messages=[
                    format_messages_prompt(prompt, {
                        'conversation': convo['prompt'],
                        'output': red_responses[convo['prompt_id']],
                        'interactions': convo['interactions']
                    }) for prompt in blue_prompts[0]
                ],
                model=args.blue_model,
                top_logprobs=5
            ) for convo in conversations
        ],
        args.out_dir + '/blue_interactions_honest_input.jsonl',
        tracker,
        'blue_interactions_honest',
        verbose=True
    )

    blue_honest_raw = read_batch_output(client, blue_honest_batch_id, args.out_dir + '/blue_interactions_honest_output.jsonl')
    blue_honest_responses = read_responses_batch(blue_honest_raw)
    blue_honest_logprobs = {result['custom_id']: result['response']['body']['choices'][0]['logprobs'] for result in blue_honest_raw}

    for convo in conversations:
        convo['red_response_0'] = red_responses[convo['prompt_id']]
        convo['blue_response_0'] = blue_honest_responses[convo['prompt_id']]
        convo['blue_logprobs_0'] = blue_honest_logprobs[convo['prompt_id']]

    with open(args.out_dir + '/conversations_honest.jsonl', 'w+') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + '\n')

    # reread tracker in case something was running in parallel    
    with open(args.tracker, 'r') as f:
        tracker = json.load(f)
    with open(args.tracker, 'w+') as f:
        json.dump(tracker, f, indent=2)

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