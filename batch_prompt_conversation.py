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
    message = conversation['prompt'] + [{
        'role': prompt['role'],
        'content': prompt['content'].replace(
            '<output>', str(previous_response)).replace('<interactions>', str(conversation['interactions']))
    }]
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
) -> str:
    with open(input_file_name, 'w+') as outfile:
        for convo in conversations:
            request = generate_request(convo, prompt, model, prev_responses)
            outfile.write(json.dumps(request) + '\n')

        batch_input_file = client.files.create(
            file=open(input_file_name, "rb"),
            purpose="batch"
        )
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

    return batch_job.id

def generate_output_batch(
    client: OpenAI,
    conversations: list[dict],
    prompt: dict[str: str],
    model: str,
    input_file_name: str,
    output_file_name: str,
    prev_responses: str | None = None,
):
    if prev_responses:
        prev_responses = read_responses_batch(prev_responses)
    batch_id = create_batch_job(client, conversations, prompt, model, input_file_name, prev_responses)
    wait_for_batch(client, batch_id)
    with open(output_file_name, 'w+') as f:
        batch_output = read_batch_output(client, batch_id)
        for response in batch_output:
            f.write(json.dumps(response) + '\n')
    return batch_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--previous_batch", type=str, default=None)
    args = parser.parse_args()

    name = os.path.basename(args.prompts).split('.')[0]
    prompts = json.load(open(args.prompts))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.conversations, 'r') as f:
        conversations = [json.loads(line) for line in f]

    for i, prompt in enumerate(prompts):
        generate_output_batch(client, conversations, prompt, args.model, f'{args.out_dir}/{name}_{i}_input.jsonl', f'{args.out_dir}/{name}_{i}_output.jsonl', args.previous_batch)