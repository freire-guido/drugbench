from batchutils import safe_update_jsonl, safe_update_json
from batchutils_multi import (
    detect_provider, get_client, generate_batch_request_multi,
    read_responses_batch_multi, read_batch_output_multi,
    run_batch_with_tracker_multi
)

from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from dotenv import load_dotenv
import argparse
import json

load_dotenv()

# Fields from the FDA label to include in RAG context (most safety-relevant)
_RAG_LABEL_FIELDS = [
    'drug_interactions',
    'contraindications',
    'warnings',
    'boxed_warning',
    'dosage_and_administration',
    'do_not_use',
]

def _combine_labels_rag(label: dict) -> str:
    parts = []
    for field in _RAG_LABEL_FIELDS:
        value = label.get(field)
        if isinstance(value, list):
            parts.append(f"## {field}\n" + "\n".join(v for v in value if isinstance(v, str)))
        elif isinstance(value, str):
            parts.append(f"## {field}\n{value}")
    return "\n\n".join(parts)


def build_fda_context(convo: dict, fda_cache: dict, max_drugs: int = 3) -> str:
    """Build FDA label context string for drugs mentioned in a conversation."""
    interactions = convo.get('interactions', {})
    if not interactions:
        return "No FDA drug label data available for this conversation."

    drug_names = list(interactions.keys())[:max_drugs]
    sections = []
    for drug in drug_names:
        cache_key = f"{drug.lower()}|3"
        entry = fda_cache.get(cache_key)
        if not entry:
            continue
        results = entry.get('data', {}).get('results', [])
        if not results:
            continue
        label_text = _combine_labels_rag(results[0])
        if label_text.strip():
            sections.append(f"### {drug}\n{label_text}")

    return "\n\n".join(sections) if sections else "No FDA drug label data available for this conversation."


def format_messages_prompt(prompt: dict, format_vars: dict) -> dict:
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

    fda_cache = json.load(open(args.fda_cache)) if args.fda_cache else {}

    blue_provider = detect_provider(args.blue_model)
    blue_client = get_client(blue_provider)

    with ThreadPoolExecutor() as executor:
        rag_0_future = executor.submit(run_batch_with_tracker_multi,
            blue_client,
            [
                generate_batch_request_multi(
                    custom_id=convo['prompt_id'],
                    messages=[
                        format_messages_prompt(prompt, {
                            'conversation': convo['prompt'],
                            'output': convo['red_response'],
                            'fda_context': build_fda_context(convo, fda_cache),
                        }) for prompt in blue_prompts[4]
                    ],
                    model=args.blue_model,
                    provider=blue_provider
                ) for convo in conversations
            ],
            args.out_dir + '/blue_rag_0_input.jsonl',
            tracker,
            'blue_rag_0',
            args.tracker,
            blue_provider,
            args.blue_model,
            True,
        )

        rag_1_future = executor.submit(run_batch_with_tracker_multi,
            blue_client,
            [
                generate_batch_request_multi(
                    custom_id=convo['prompt_id'],
                    messages=[
                        format_messages_prompt(prompt, {
                            'conversation': convo['prompt'],
                            'output': convo['red_response'],
                            'fda_context': build_fda_context(convo, fda_cache),
                        }) for prompt in blue_prompts[5]
                    ],
                    model=args.blue_model,
                    provider=blue_provider
                ) for convo in conversations
            ],
            args.out_dir + '/blue_rag_1_input.jsonl',
            tracker,
            'blue_rag_1',
            args.tracker,
            blue_provider,
            args.blue_model,
            True,
        )

    rag_batch_ids = [rag_0_future.result(), rag_1_future.result()]
    rag_responses = []
    for i, batch_id in enumerate(rag_batch_ids):
        lines, custom_id_mapping = read_batch_output_multi(
            blue_client, batch_id,
            args.out_dir + f'/blue_rag_{i}_output.jsonl',
            blue_provider
        )
        rag_responses.append(read_responses_batch_multi(lines, blue_provider, custom_id_mapping))

    for convo in conversations:
        convo['blue_rag_0'] = rag_responses[0][convo['prompt_id']]
        convo['blue_rag_1'] = rag_responses[1][convo['prompt_id']]

    safe_update_jsonl(args.conversations, conversations)
    safe_update_json(args.tracker, tracker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, required=True)
    parser.add_argument("--blue_prompts", type=str, required=True)
    parser.add_argument("--blue_model", type=str, default="gpt-4o")
    parser.add_argument("--fda_cache", type=str, default=None,
                        help="Path to openfda_cache.json")
    parser.add_argument("--tracker", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default='')
    args = parser.parse_args()
    main(args)
