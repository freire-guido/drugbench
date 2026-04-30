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

def format_messages_prompt(prompt, format_vars):
    return {"role": prompt['role'], "content": prompt['content'].format(**format_vars)}

def apply_placeholdium(text, act):
    return (text
        .replace("Placeholdium's", act + "'s")
        .replace("Placeholdium", act)
        .replace("placeholdium's", act + "'s")
        .replace("placeholdium", act)
        .split('<explanation>')[0].strip()
    )

def main(args):
    with open(args.conversations) as f:
        conversations = [json.loads(line) for line in f]
    with open(args.red_prompts) as f:
        red_prompts = json.load(f)
    with open(args.blue_prompts) as f:
        blue_prompts = json.load(f)
    with open(args.tracker) as f:
        tracker = json.load(f)

    red_provider = detect_provider(args.red_model)
    blue_provider = detect_provider(args.blue_model)
    red_client = get_client(red_provider)
    blue_client = get_client(blue_provider)

    # Phase 1: N red resamples in parallel
    def make_red_requests():
        return [
            generate_batch_request_multi(
                custom_id=c['prompt_id'],
                messages=red_prompts[args.prompt_index] + c['prompt'],
                model=args.red_model,
                provider=red_provider
            ) for c in conversations
        ]

    with ThreadPoolExecutor() as executor:
        red_futures = {
            i: executor.submit(
                run_batch_with_tracker_multi,
                red_client, make_red_requests(),
                f"{args.out_dir}/red_resample_{i}_input.jsonl",
                tracker, f"red_resample_{i}", args.tracker,
                red_provider, args.red_model, True
            ) for i in range(args.n)
        }

    red_responses = {}
    for i, future in red_futures.items():
        lines, mapping = read_batch_output_multi(
            red_client, future.result(),
            f"{args.out_dir}/red_resample_{i}_output.jsonl", red_provider
        )
        raw = read_responses_batch_multi(lines, red_provider, mapping)
        red_responses[i] = {}
        for c in conversations:
            text = raw[c['prompt_id']]
            act = c.get('red_0_act', '')
            red_responses[i][c['prompt_id']] = apply_placeholdium(text, act) if act else text

    # Phase 2: harm score (blue_prompts[0]) for each resample, all in parallel
    with ThreadPoolExecutor() as executor:
        blue_futures = {
            i: executor.submit(
                run_batch_with_tracker_multi,
                blue_client,
                [
                    generate_batch_request_multi(
                        custom_id=c['prompt_id'],
                        messages=[format_messages_prompt(p, {
                            'conversation': c['prompt'],
                            'output': red_responses[i][c['prompt_id']]
                        }) for p in blue_prompts[0]],
                        model=args.blue_model,
                        provider=blue_provider
                    ) for c in conversations
                ],
                f"{args.out_dir}/blue_resample_{i}_input.jsonl",
                tracker, f"blue_resample_{i}", args.tracker,
                blue_provider, args.blue_model, True
            ) for i in range(args.n)
        }

    blue_responses = {}
    for i, future in blue_futures.items():
        lines, mapping = read_batch_output_multi(
            blue_client, future.result(),
            f"{args.out_dir}/blue_resample_{i}_output.jsonl", blue_provider
        )
        blue_responses[i] = read_responses_batch_multi(lines, blue_provider, mapping)

    for c in conversations:
        pid = c['prompt_id']
        c['red_resample'] = [red_responses[i][pid] for i in range(args.n)]
        c['blue_resample_0'] = [blue_responses[i][pid] for i in range(args.n)]

    safe_update_jsonl(args.conversations, conversations)
    safe_update_json(args.tracker, tracker)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", required=True)
    parser.add_argument("--red_prompts", required=True)
    parser.add_argument("--blue_prompts", required=True)
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--red_model", default="gpt-4o")
    parser.add_argument("--blue_model", default="gpt-4o")
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--out_dir", default='')
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args()
    main(args)
