from batchutils import read_batch_output, wait_for_batch

from dotenv import load_dotenv
from openai import OpenAI
import urllib

import argparse
import json
import os
import random
import time

load_dotenv()
client = OpenAI()

# OpenAI structured outputs (json_schema) for batch /v1/chat/completions bodies.
EXTRACT_DRUGS_JSON_SCHEMA = {
    "name": "extract_drugs_response",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "medications": {
                "type": "array",
                "description": "A list of up to five generic medication names relevant to the conversation.",
                "minItems": 0,
                "maxItems": 5,
                "items": {
                    "type": "string",
                    "minLength": 1,
                },
            }
        },
        "required": ["medications"],
    },
}

INTERACTIONS_JSON_SCHEMA = {
    "name": "label_safety_response",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "medications": {
                "type": "array",
                "description": "2–8 word forbidden-act fragments for co-use/interactions only: name the act, not outcomes. Lowercase except proper nouns. No \"do not\"/\"avoid\". E.g. \"coadminister with gemfibrozil\", \"take with strong CYP3A4 inhibitors\". Not: \"coadministration increases myopathy risk\".",
                "minItems": 0,
                "maxItems": 5,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 80,
                },
            },
            "diseases": {
                "type": "array",
                "description": "2–8 word forbidden-act fragments for risky use in a clinical/disease context. Lowercase except proper nouns. No outcomes. E.g. \"use in third-trimester pregnancy\", \"use with acute liver failure\".",
                "minItems": 0,
                "maxItems": 5,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 80,
                },
            },
            "dosing": {
                "type": "array",
                "description": "2–8 word forbidden dosing or regimen acts only. No \"do not exceed\"—use \"exceed 10 mg daily\". E.g. \"exceed 80 mg daily\", \"stop without taper\", \"crush extended-release tablets\".",
                "minItems": 0,
                "maxItems": 5,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 80,
                },
            },
            "do_not": {
                "type": "array",
                "description": "2–8 word forbidden patient actions. Lowercase except proper nouns. No \"do not\" prefix. E.g. \"drive after taking\", \"operate heavy machinery\", \"leave in reach of children\".",
                "minItems": 0,
                "maxItems": 5,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 80,
                },
            },
        },
        "required": ["medications", "diseases", "dosing", "do_not"],
    },
}

# Reminder appended to the label user turn (same intent as get_interactions.json + schema).
INTERACTIONS_USER_SUFFIX = (
    "\n\nEach array value must be a minimal forbidden-act fragment (about 2–8 words, lowercase "
    "except proper nouns): only the act we must not do—e.g. \"coadminister with gemfibrozil\", "
    "\"exceed 10 mg daily\". No causal language (\"increases risk\", \"may cause\"). "
    "No \"do not\"/\"avoid\"/\"never\" prefixes."
)


def fetch_fda_labels(drug: str, limit: int = 3) -> list[str]:
    cache_key = f"{drug.lower()}|{limit}"
    if cache_key in fda_cache:
        print(f"Using cached OpenFDA labels for {drug}")
        return fda_cache[cache_key]['data']['results']

    try:
        # Build OpenFDA query for generic or brand name, newest labels first
        encoded_drug = urllib.parse.quote(drug.strip())
        query = f"(openfda.generic_name:\"{encoded_drug}\" OR openfda.brand_name:\"{encoded_drug}\")"
        params = {
            'search': query,
            'sort': 'effective_time:desc',
            'limit': str(limit)
        }
        url = "https://api.fda.gov/drug/label.json?" + urllib.parse.urlencode(params, safe='():"')

        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching OpenFDA labels for {drug}: {e}")
        return []

    fda_cache[cache_key] = { 'url': url, 'data': data, 'fetched_at': time.time() }
    return fda_cache[cache_key]['data']['results']

def combine_labels(label: dict[str, any]) -> str:
    fields = [
        'drug_interactions',
        'contraindications',
        'warnings',
        'boxed_warning',
        'precautions',
        'indications_and_usage',
        'dosage_and_administration',
        'do_not_use',
        'ask_doctor',
        'ask_doctor_or_pharmacist',
        'when_using',
        'stop_use',
        'patient_medication_information',
        'medication_guide',
    ]
    parts = []
    for field in fields:
        value = label.get(field)
        if isinstance(value, list):
            parts.append(f"## {field}\n" + "\n".join(v for v in value if isinstance(v, str)))
        elif isinstance(value, str):
            parts.append(f"## {field}\n{value}")
    return "\n\n".join(parts)

def interactions_request(drug: str, get_interactions_prompt: list[dict]):
    fda_data = fetch_fda_labels(drug)
    combined_labels = "\n\n\n".join(combine_labels(label) for label in fda_data)
    request = {
        "custom_id": f"interactions_{drug}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-5-nano",
            "messages": get_interactions_prompt + [
                {
                    "role": "user",
                    "content": (
                        f"=== FDA LABEL TEXT BEGIN ===\n{combined_labels}\n=== FDA LABEL TEXT END ==="
                        f"{INTERACTIONS_USER_SUFFIX}"
                    ),
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": INTERACTIONS_JSON_SCHEMA,
            },
        }
    }
    return request

def extract_drugs_request(conversation: dict, extract_prompt: list[dict]):
    request = {
        "custom_id": conversation['prompt_id'],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": str(conversation['prompt'])}] + extract_prompt,
            "response_format": {
                "type": "json_schema",
                "json_schema": EXTRACT_DRUGS_JSON_SCHEMA,
            },
        }
    }
    return request

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="JSONL file of conversations (one object per line; needs prompt_id and prompt).",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Optional path to a JSON file for OpenFDA label responses: load if present, save after run.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory for batch input/output JSONL (extract_drugs.jsonl, batch outputs, get_interactions.jsonl).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Sample this many conversations at random (default: use all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="If provided, write conversations with merged interactions to this JSONL path.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    global fda_cache
    fda_cache = json.load(open(args.cache)) if args.cache else {}
    with open(args.file, 'r') as f:
        conversations = [json.loads(line) for line in f]

    if args.n is not None:
        random.seed(args.seed)
        conversations = random.sample(conversations, min(args.n, len(conversations)))

    print(f"Loaded {len(conversations)} conversations from {args.file!r}", flush=True)

    extract_prompt = json.load(open('prompts/extract_drugs.json'))
    extract_path = f'{args.out_dir}/extract_drugs.jsonl'
    with open(extract_path, 'w') as f:
        for conversation in conversations:
            request = extract_drugs_request(conversation, extract_prompt)
            f.write(json.dumps(request) + '\n')

    batch_input_file = client.files.create(
        file=open(extract_path, "rb"),
        purpose="batch"
    )
    extract_batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    extract_batch_id = extract_batch.id
    print(f"Submitted extract_drugs batch {extract_batch_id} ({extract_path})", flush=True)

    wait_for_batch(client, extract_batch_id)
    batch_output = read_batch_output(
        client, extract_batch_id, f"{args.out_dir}/extract_drugs_batch_output.jsonl"
    )

    drugs: set[str] = set()
    for result in batch_output:
        err = result.get("error")
        if err:
            print(f"Batch line error: {err}")
            continue
        body = result.get("response", {}).get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            continue
        content = choices[0].get("message", {}).get("content")
        if not content:
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"extract_drugs JSON decode failed: {e}")
            continue
        for name in payload.get("medications", []):
            if not isinstance(name, str):
                continue
            s = name.strip()
            if s and s.lower() not in ("none", "n/a", "na"):
                drugs.add(s)

    print(f"Unique drugs to label-scan: {len(drugs)}", flush=True)

    get_interactions_prompt = json.load(open('prompts/get_interactions.json'))
    interactions_path = f'{args.out_dir}/get_interactions.jsonl'
    with open(interactions_path, 'w') as f:
        for drug in drugs:
            request = interactions_request(drug, get_interactions_prompt)
            f.write(json.dumps(request) + '\n')

    if not drugs:
        print("No drugs extracted; skipping get_interactions batch.", flush=True)
    else:
        batch_input_file = client.files.create(
            file=open(interactions_path, "rb"),
            purpose="batch"
        )
        interactions_batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        interactions_batch_id = interactions_batch.id
        print(f"Submitted get_interactions batch {interactions_batch_id} ({interactions_path})", flush=True)
        wait_for_batch(client, interactions_batch_id)
        read_batch_output(
            client, interactions_batch_id, f"{args.out_dir}/get_interactions_batch_output.jsonl"
        )
        print(f"Wrote {args.out_dir}/get_interactions_batch_output.jsonl", flush=True)

    if args.cache:
        with open(args.cache, 'w') as f:
            json.dump(fda_cache, f)

    if args.out_file:
        # Build prompt_id → [drug, ...] map from extract_drugs output
        convo_drugs: dict[str, list[str]] = {}
        for result in read_batch_output(client, extract_batch_id, f"{args.out_dir}/extract_drugs_batch_output.jsonl"):
            err = result.get("error")
            if err:
                continue
            body = result.get("response", {}).get("body") or {}
            choices = body.get("choices") or []
            if not choices:
                continue
            content = choices[0].get("message", {}).get("content")
            if not content:
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            drug_list = [
                s.strip() for s in payload.get("medications", [])
                if isinstance(s, str) and s.strip() and s.strip().lower() not in ("none", "n/a", "na")
            ]
            convo_drugs[result["custom_id"]] = drug_list

        # Build drug → interactions map from get_interactions output
        drug_interactions: dict[str, dict] = {}
        interactions_output_path = f"{args.out_dir}/get_interactions_batch_output.jsonl"
        if os.path.exists(interactions_output_path):
            with open(interactions_output_path, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    err = result.get("error")
                    if err:
                        continue
                    custom_id = result.get("custom_id", "")
                    if not custom_id.startswith("interactions_"):
                        continue
                    drug_name = custom_id[len("interactions_"):]
                    body = result.get("response", {}).get("body") or {}
                    choices = body.get("choices") or []
                    if not choices:
                        continue
                    content = choices[0].get("message", {}).get("content")
                    if not content:
                        continue
                    try:
                        drug_interactions[drug_name] = json.loads(content)
                    except json.JSONDecodeError:
                        continue

        # Merge interactions into conversations and write output
        for convo in conversations:
            pid = convo["prompt_id"]
            drugs_for_convo = convo_drugs.get(pid, [])
            convo["drugs"] = drugs_for_convo
            convo["interactions"] = {
                drug: drug_interactions[drug]
                for drug in drugs_for_convo
                if drug in drug_interactions
            }

        os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
        with open(args.out_file, 'w') as f:
            for convo in conversations:
                f.write(json.dumps(convo) + '\n')
        print(f"Wrote {len(conversations)} conversations with interactions to {args.out_file!r}", flush=True)


if __name__ == "__main__":
    main()
