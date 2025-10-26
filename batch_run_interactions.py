from batchutils import read_batch_output, wait_for_batch

from dotenv import load_dotenv
from openai import OpenAI

import argparse
import json

load_dotenv()
client = OpenAI()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        conversations = [json.loads(line) for line in f]

    extract_prompt = json.load(open('prompts/extract_drugs.json'))
    for conversation in conversations:
        request = {
            "custom_id": conversation['prompt_id'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5",
                "messages": {"role": "user", "content": str(conversation['prompt'])} + extract_prompt
            }
        }
        with open(f'{args.out_dir}/extract_drugs.jsonl', 'w') as f:
            f.write(json.dumps(request) + '\n')

    batch_input_file = client.files.create(
        file=open(f'{args.out_dir}/extract_drugs.jsonl', "rb"),
        purpose="batch"
    )
    batch_id = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    wait_for_batch(batch_id)
    batch_output = read_batch_output(batch_id)
    
    