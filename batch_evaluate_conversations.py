from logging import exception
from batchutils import wait_for_batch, read_batch_output

from dotenv import load_dotenv
from openai import OpenAI
import argparse
import json

load_dotenv()
client = OpenAI()

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

def generate_evaluation_requests(conversation: dict, model: str, key: str,step: str):
    convo_with_response = conversation['prompt'] + [{'role': 'assistant', 'content': conversation[key][step]}]
    convo_str = "\n\n".join(
        [f"{m['role']}: {m['content']}" for m in convo_with_response]
    )
    requests = []
    for i, rubric_item in enumerate(conversation['rubrics']):
        grader_prompt = GRADER_TEMPLATE.replace(
            "<<conversation>>", convo_str
        ).replace("<<rubric_item>>", str(rubric_item))
        request = {
            "custom_id": conversation['prompt_id'] + f'_{i}',
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": grader_prompt}]
            }
        }
        requests.append(request)
    return requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--step", type=str, required=True)
    args = parser.parse_args()

    with open(args.dir + '/conversations.jsonl', 'r') as f:
        conversations = [json.loads(line) for line in f]

    requests = []
    for conversation in conversations:
        try:
            reqs = generate_evaluation_requests(conversation, args.model, 'blue_responses', args.step)
        except Exception as e:
            print(f"Error generating evaluation request for conversation {conversation['prompt_id']}: {e}")
            continue
        requests.extend(reqs)
    with open(f'batch/blue_evaluations_{args.step}.jsonl', 'w+') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')

    print(f"Creating blue batch job")
    batch_input_file = client.files.create(
        file=open(f'batch/blue_evaluations_{args.step}.jsonl', "rb"),
        purpose="batch"
    )
    blue_batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    requests = []
    for conversation in conversations:
        try:
            reqs = generate_evaluation_requests(conversation, args.model, 'red_responses', args.step)
        except Exception as e:
            print(f"Error generating evaluation request for conversation {conversation['prompt_id']}: {e}")
            continue
        requests.extend(reqs)
    with open(f'batch/red_evaluations_{args.step}.jsonl', 'w+') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')

    print(f"Creating red batch job")
    batch_input_file = client.files.create(
        file=open(f'batch/red_evaluations_{args.step}.jsonl', "rb"),
        purpose="batch"
    )
    red_batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"Waiting for blue batch job {blue_batch_job.id} to complete")
    wait_for_batch(client, blue_batch_job.id)
    print(f"Waiting for red batch job {red_batch_job.id} to complete")
    wait_for_batch(client, red_batch_job.id)

    blue_batch_output = read_batch_output(client, blue_batch_job.id)
    blue_batch_output = {res['custom_id']: res['response']['body']['choices'][0]['message']['content'] for res in blue_batch_output}
    red_batch_output = read_batch_output(client, red_batch_job.id)
    red_batch_output = {res['custom_id']: res['response']['body']['choices'][0]['message']['content'] for res in red_batch_output}

    print(f"Saving evaluations to {args.dir + f'/evaluations_{args.step}.jsonl'}")
    with open(args.dir + f'/evaluations_{args.step}.jsonl', 'w+') as f:
        ids = set(blue_batch_output.keys()) & set(red_batch_output.keys())
        for id in ids:
            f.write(json.dumps({'prompt_id': id, 'blue_evaluation': blue_batch_output[id], 'red_evaluation': red_batch_output[id]}) + '\n')

