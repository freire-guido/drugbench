from batchutils import wait_for_batch, read_batch_output, read_responses_batch

from dotenv import load_dotenv
from openai import OpenAI
import argparse
import json
import os

load_dotenv()
client = OpenAI()

def generate_request(conversation: list[dict], prompt: dict[str: str], model: str, prev_responses: dict[str: str] | None = None) -> dict:
    if prev_responses and conversation['prompt_id'] not in prev_responses:
        print(f"Warning: Prompt ID {conversation['prompt_id']} not found in previous responses")
        previous_response = None
    elif not prev_responses:
        previous_response = None
    else:
        previous_response = prev_responses[conversation['prompt_id']]
    message = [{
        'role': prompt['role'],
        'content': prompt['content']
            .replace('<output>', str(previous_response))
            .replace('<interactions>', str(conversation['interactions']))
    }] + conversation['prompt']
    request = {
        "custom_id": conversation['prompt_id'],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": message
        }
    }
    return request


def create_batch_job(
    client: OpenAI,
    conversations: list[dict],
    prompt: dict[str: str],
    model: str,
    input_file_name: str,
    prev_responses: dict[str: str] | None = None,
    verbose: bool = False,
) -> str:
    if verbose:
        print(f"Creating batch job for {len(conversations)} conversations...")
    with open(input_file_name, 'w+') as outfile:
        for convo in conversations:
            request = generate_request(convo, prompt, model, prev_responses)
            outfile.write(json.dumps(request) + '\n')

    batch_input_file = client.files.create(
        file=open(input_file_name, "rb"),
        purpose="batch"
    )
    if verbose:
        print(f"Uploaded input file: {batch_input_file.id}")
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    if verbose:
        print(f"Created batch job: {batch_job.id}")

    return batch_job.id

def generate_output_batch(
    client: OpenAI,
    conversations: list[dict],
    prompt: dict[str: str],
    model: str,
    input_file_name: str,
    output_file_name: str,
    prev_responses = None,
    verbose: bool = False,
):
    if isinstance(prev_responses, str):
        if verbose:
            print(f"Loading previous responses from {prev_responses}...")
        prev_responses = read_responses_batch(open(prev_responses, 'r'))
    elif isinstance(prev_responses, list):
        if verbose:
            print(f"Loading previous responses from list of {len(prev_responses)} responses...")
        prev_responses = read_responses_batch(prev_responses)
    batch_id = create_batch_job(client, conversations, prompt, model, input_file_name, prev_responses, verbose)
    if verbose:
        print(f"Waiting for batch {batch_id} to complete...")
    wait_for_batch(client, batch_id)
    if verbose:
        print(f"Batch completed, reading output...")
    with open(output_file_name, 'w+') as f:
        batch_output = read_batch_output(client, batch_id)
        for response in batch_output:
            f.write(json.dumps(response) + '\n')
    if verbose:
        print(f"Saved {len(batch_output)} responses to {output_file_name}")

    return batch_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--previous_batch", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Enable verbose output")
    args = parser.parse_args()

    name = os.path.basename(args.prompts).split('.')[0]
    prompts = json.load(open(args.prompts))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.conversations, 'r') as f:
        conversations = [json.loads(line) for line in f]
    
    if args.verbose:
        print(f"Processing {len(prompts)} prompt(s) for {len(conversations)} conversation(s)")
        print(f"Model: {args.model}")
        print(f"Output directory: {args.out_dir}")

    for i, prompt in enumerate(prompts):
        if args.verbose:
            print(f"\nProcessing prompt {i+1}/{len(prompts)}...")
        generate_output_batch(client, conversations, prompt, args.model, f'{args.out_dir}/{name}_{i}_input.jsonl', f'{args.out_dir}/{name}_{i}_output.jsonl', args.previous_batch, args.verbose)
    
    if args.verbose:
        print("\nCompleted")