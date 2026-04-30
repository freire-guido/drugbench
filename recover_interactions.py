"""
Recovery script: complete partial get_interactions.jsonl, submit batch, and merge into final JSONL.
Usage:
  # Resume from scratch (fetch missing drugs + submit batch):
  python recover_interactions.py --out_file datasets/healthbench_consensus_interactions.jsonl

  # Skip fetch+submit if batch already completed:
  python recover_interactions.py --batch_id <id> --out_file datasets/healthbench_consensus_interactions.jsonl
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

from dotenv import load_dotenv
from openai import OpenAI

from batchutils import read_batch_output, wait_for_batch
from batch_run_interactions import (
    INTERACTIONS_JSON_SCHEMA,
    INTERACTIONS_USER_SUFFIX,
    combine_labels,
    interactions_request,
)

load_dotenv()
client = OpenAI()

OUT_DIR = "/tmp/healthbench_consensus_interactions_run"
EXTRACT_BATCH_OUTPUT = f"{OUT_DIR}/extract_drugs_batch_output.jsonl"
INTERACTIONS_INPUT = f"{OUT_DIR}/get_interactions.jsonl"
INTERACTIONS_BATCH_OUTPUT = f"{OUT_DIR}/get_interactions_batch_output.jsonl"
CONVERSATIONS_FILE = "datasets/healthbench_consensus.jsonl"
CACHE_FILE = "datasets/openfda_cache.json"


def get_all_drugs_from_extract_output() -> set[str]:
    drugs = set()
    with open(EXTRACT_BATCH_OUTPUT) as f:
        for line in f:
            result = json.loads(line)
            if result.get("error"):
                continue
            choices = (result.get("response", {}).get("body") or {}).get("choices") or []
            if not choices:
                continue
            content = choices[0].get("message", {}).get("content")
            if not content:
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            for s in payload.get("medications", []):
                if isinstance(s, str):
                    s = s.strip()
                    if s and s.lower() not in ("none", "n/a", "na"):
                        drugs.add(s)
    return drugs


def get_already_fetched_drugs() -> set[str]:
    fetched = set()
    with open(INTERACTIONS_INPUT) as f:
        for line in f:
            r = json.loads(line)
            cid = r.get("custom_id", "")
            if cid.startswith("interactions_"):
                fetched.add(cid[len("interactions_"):])
    return fetched


def fetch_fda_labels(drug: str, cache: dict, limit: int = 3) -> list:
    cache_key = f"{drug.lower()}|{limit}"
    if cache_key in cache:
        return cache[cache_key].get("data", {}).get("results", [])
    try:
        encoded = urllib.parse.quote(drug.strip())
        query = f'(openfda.generic_name:"{encoded}" OR openfda.brand_name:"{encoded}")'
        params = {"search": query, "sort": "effective_time:desc", "limit": str(limit)}
        url = "https://api.fda.gov/drug/label.json?" + urllib.parse.urlencode(params, safe='():"')
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cache[cache_key] = {"url": url, "data": data, "fetched_at": time.time()}
        return data.get("results", [])
    except Exception as e:
        print(f"  OpenFDA 404/error for {drug!r}: {e}")
        return []


def complete_interactions_jsonl(missing_drugs: set[str], cache: dict) -> None:
    get_interactions_prompt = json.load(open("prompts/get_interactions.json"))
    appended = 0
    with open(INTERACTIONS_INPUT, "a") as f:
        for i, drug in enumerate(sorted(missing_drugs), 1):
            print(f"  [{i}/{len(missing_drugs)}] Fetching FDA labels for {drug!r}", flush=True)
            fda_data = fetch_fda_labels(drug, cache)
            combined = "\n\n\n".join(combine_labels(label) for label in fda_data)
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
                                f"=== FDA LABEL TEXT BEGIN ===\n{combined}\n=== FDA LABEL TEXT END ==="
                                f"{INTERACTIONS_USER_SUFFIX}"
                            ),
                        }
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": INTERACTIONS_JSON_SCHEMA,
                    },
                },
            }
            f.write(json.dumps(request) + "\n")
            appended += 1
    print(f"Appended {appended} missing drug requests to {INTERACTIONS_INPUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_id", default=None, help="Completed get_interactions batch ID (skips fetch+submit)")
    parser.add_argument("--out_file", required=True, help="Output JSONL path")
    args = parser.parse_args()

    if args.batch_id:
        batch = client.batches.retrieve(args.batch_id)
        print(f"Batch {args.batch_id} status: {batch.status}")
        if batch.status != "completed":
            wait_for_batch(client, args.batch_id)
        batch_id = args.batch_id
    else:
        # Find and fetch missing drugs
        all_drugs = get_all_drugs_from_extract_output()
        fetched = get_already_fetched_drugs()
        missing = all_drugs - fetched
        print(f"Total unique drugs: {len(all_drugs)}, already fetched: {len(fetched)}, missing: {len(missing)}")

        if missing:
            cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}
            complete_interactions_jsonl(missing, cache)
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)
            print(f"Saved updated cache to {CACHE_FILE}")

        print(f"Submitting get_interactions batch ({INTERACTIONS_INPUT}) ...")
        batch_input_file = client.files.create(file=open(INTERACTIONS_INPUT, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_id = batch.id
        print(f"Submitted batch {batch_id}, waiting for completion...")
        wait_for_batch(client, batch_id)

    read_batch_output(client, batch_id, INTERACTIONS_BATCH_OUTPUT)
    print(f"Downloaded get_interactions output to {INTERACTIONS_BATCH_OUTPUT}")

    # Build prompt_id → drugs map
    convo_drugs: dict[str, list[str]] = {}
    with open(EXTRACT_BATCH_OUTPUT) as f:
        for line in f:
            result = json.loads(line)
            if result.get("error"):
                continue
            choices = (result.get("response", {}).get("body") or {}).get("choices") or []
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

    # Build drug → interactions map
    drug_interactions: dict[str, dict] = {}
    with open(INTERACTIONS_BATCH_OUTPUT) as f:
        for line in f:
            result = json.loads(line)
            if result.get("error"):
                continue
            custom_id = result.get("custom_id", "")
            if not custom_id.startswith("interactions_"):
                continue
            drug_name = custom_id[len("interactions_"):]
            choices = (result.get("response", {}).get("body") or {}).get("choices") or []
            if not choices:
                continue
            content = choices[0].get("message", {}).get("content")
            if not content:
                continue
            try:
                drug_interactions[drug_name] = json.loads(content)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(convo_drugs)} convo→drug mappings, {len(drug_interactions)} drug interaction records")

    with open(CONVERSATIONS_FILE) as f:
        conversations = [json.loads(line) for line in f]

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
    with open(args.out_file, "w") as f:
        for convo in conversations:
            f.write(json.dumps(convo) + "\n")
    print(f"Wrote {len(conversations)} conversations to {args.out_file!r}")


if __name__ == "__main__":
    main()
