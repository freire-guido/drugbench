from batchutils import read_batch_output, wait_for_batch

from dotenv import load_dotenv
from openai import OpenAI
import urllib

import argparse
import json
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
                "description": "Generic medication names relevant to the conversation, at most five.",
                "items": {"type": "string"},
                "maxItems": 4,
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
                "description": "Other drugs, classes, or substances to avoid or use with caution.",
                "items": {"type": "string"},
                "maxItems": 4,
            },
            "diseases": {
                "type": "array",
                "description": "Conditions where use is contraindicated or requires caution.",
                "items": {"type": "string"},
                "maxItems": 4,
            },
            "do_not": {
                "type": "array",
                "description": "Actions the patient should not do; short phrases (e.g. drink alcohol).",
                "items": {"type": "string"},
                "maxItems": 4,
            },
        },
        "required": ["medications", "diseases", "do_not"],
    },
}


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
            "messages": get_interactions_prompt + [{"role": "user", "content": f"=== FDA LABEL TEXT BEGIN ===\n{combined_labels}\n=== FDA LABEL TEXT END ==="}],
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
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    global fda_cache
    fda_cache = json.load(open(args.cache)) if args.cache else {}
    with open(args.file, 'r') as f:
        conversations = [json.loads(line) for line in f]

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
    batch_id = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    wait_for_batch(client, batch_id)
    batch_output = read_batch_output(
        client, batch_id, f"{args.out_dir}/extract_drugs_batch_output.jsonl"
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

    get_interactions_prompt = json.load(open('prompts/get_interactions.json'))
    interactions_path = f'{args.out_dir}/get_interactions.jsonl'
    with open(interactions_path, 'w') as f:
        for drug in drugs:
            request = interactions_request(drug, get_interactions_prompt)
            f.write(json.dumps(request) + '\n')

    batch_input_file = client.files.create(
        file=open(interactions_path, "rb"),
        purpose="batch"
    )
    batch_id = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    if args.cache:
        with open(args.cache, 'w') as f:
            json.dump(fda_cache, f)
