from dotenv import load_dotenv
from openai import OpenAI
import argparse
import json

load_dotenv()
client = OpenAI()

def generate_request(conversation: list[dict], prompt: dict[str: str], prev_responses: dict[str: str] | None = None) -> dict:
    if prev_responses and conversation['prompt_id'] not in prev_responses:
        print(f"Warning: Prompt ID {conversation['prompt_id']} not found in previous responses")
    previous_response = prev_responses[conversation['prompt_id']] if prev_responses else None
    message = conversation['prompt'] + {
        'role': prompt['role'],
        'content': prompt['content'].replace('<output>', previous_response)
    }
    request = {
        "custom_id": conversation['prompt_id'],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-5",
            "messages": [message]
        }
    }
    return request

def read_previous_batch(file: str) -> dict:
    with open(file, 'r') as f:
        res = [json.loads(line) for line in f]
        return {res['custom_id']: res['response']['body']['choices'][0]['message']['content'] for res in res}

def main(conversations: list[dict], prompt: dict[str: str], step: int, prev_responses: dict[str: str] | None = None):
    with open(f'batch/blue_team_{step}.jsonl', 'w') as outfile:
        for convo in conversations:
            request = generate_request(convo, prompt, prev_responses)
            outfile.write(json.dumps(request) + '\n')

        batch_input_file = client.files.create(
            file=open(f'batch/blue_team_{step}.jsonl', "rb"),
            purpose="batch"
        )
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

    return batch_job['id']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--tracker", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    prompts = json.load(open(args.prompts))
    tracker = json.load(open(args.tracker))
    with open(args.file, 'r') as f:
        conversations = [json.loads(line) for line in f]

    if args.step > 0:
        prev_batch_file = f'batch/{tracker["blue_team"][args.step-1]}_output.jsonl'
        prev_responses = read_previous_batch(prev_batch_file)
    
    id = main(conversations, prompts[args.step], args.step, prev_responses)

    tracker['blue_team'][args.step] = id
    with open(tracker, 'w') as f:
        json.dump(tracker, f) 