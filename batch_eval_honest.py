from batchutils import safe_update_jsonl, safe_update_json
from batchutils_multi import (
    detect_provider, get_client, generate_batch_request_multi,
    read_responses_batch_multi, read_batch_output_multi,
    run_batch_with_tracker_multi
)

from dotenv import load_dotenv
import argparse
import json

load_dotenv()

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
{conversation}

# Rubric item
{rubric_item}

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

def main(args):
    with open(args.conversations, 'r') as f:
        conversations = [json.loads(line) for line in f]
    with open(args.tracker, 'r') as f:
        tracker = json.load(f)

    # Detect provider and get client
    eval_provider = detect_provider(args.model)
    eval_client = get_client(eval_provider)

    eval_honest_batch_id = run_batch_with_tracker_multi(
        eval_client,
        [
            generate_batch_request_multi(
                custom_id=convo['prompt_id'] + f'_{i}',
                messages=[{
                    'role': 'user',
                    'content': GRADER_TEMPLATE.format(
                        conversation=str(convo['prompt'] + [{'role': 'assistant', 'content': convo[args.response_field]}]),
                        rubric_item=str(rubric_item)
                    )
                }],
                model=args.model,
                provider=eval_provider
            ) for convo in conversations for i, rubric_item in enumerate(convo['rubrics'])
        ],
        args.out_dir + f'/eval_{args.response_field}_honest_input.jsonl',
        tracker,
        f'eval_{args.response_field}_honest',
        args.tracker,
        eval_provider,
        args.model,
        True,
    )

    lines, custom_id_mapping = read_batch_output_multi(eval_client, eval_honest_batch_id, args.out_dir + f'/eval_{args.response_field}_honest_output.jsonl', eval_provider)
    eval_honest_responses = read_responses_batch_multi(lines, eval_provider, custom_id_mapping)

    missing_eval_count = 0
    eval_field = f'eval_{args.response_field}_honest'
    for convo in conversations:
        # Older conversation files may include this field as null; normalize to a list.
        if not isinstance(convo.get(eval_field), list):
            convo[eval_field] = []
        for i, _ in enumerate(convo['rubrics']):
            response_key = convo['prompt_id'] + f'_{i}'
            response_value = eval_honest_responses.get(response_key)
            if response_value is None:
                missing_eval_count += 1
            convo[eval_field].append(response_value)

    if missing_eval_count:
        print(
            f"Warning: missing {missing_eval_count} eval responses for "
            f"{args.response_field}; appended null placeholders."
        )

    safe_update_jsonl(args.conversations, conversations)
    safe_update_json(args.tracker, tracker)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, required=True)
    parser.add_argument("--response_field", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--tracker", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default='')
    args = parser.parse_args()
    main(args)