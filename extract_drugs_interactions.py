import json, time
import urllib.parse, urllib.request
from typing import Dict, Any, List

from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENFDA_CACHE_PATH = 'openfda_cache.json'

def _read_cache() -> Dict[str, Any]:
    try:
        with open(OPENFDA_CACHE_PATH, 'r') as infile:
            return json.load(infile)
    except Exception:
        return {}

def _write_cache(cache: Dict[str, Any]) -> None:
    try:
        with open(OPENFDA_CACHE_PATH, 'w') as outfile:
            json.dump(cache, outfile)
    except Exception:
        pass

def _fetch_openfda_labels(drug: str, limit: int = 3) -> Dict[str, Any]:
    cache = _read_cache()
    cache_key = f"{drug.lower()}|{limit}"
    if cache_key in cache:
        return cache[cache_key]

    # Build OpenFDA query for generic or brand name, newest labels first
    encoded_drug = urllib.parse.quote(drug.strip())
    query = f"(openfda.generic_name:\"{encoded_drug}\" OR openfda.brand_name:\"{encoded_drug}\")"
    params = {
        'search': query,
        'sort': 'effective_time:desc',
        'limit': str(limit),
    }
    url = "https://api.fda.gov/drug/label.json?" + urllib.parse.urlencode(params, safe='():"')

    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode('utf-8'))

    cache[cache_key] = { 'url': url, 'data': data, 'fetched_at': time.time() }
    _write_cache(cache)
    return cache[cache_key]

def _combine_label_texts(label: Dict[str, Any]) -> str:
    # Collect relevant textual sections that mention interactions, contraindications, and warnings
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
    parts: List[str] = []
    for field in fields:
        value = label.get(field)
        if isinstance(value, list):
            parts.append(f"## {field}\n" + "\n".join(v for v in value if isinstance(v, str)))
        elif isinstance(value, str):
            parts.append(f"## {field}\n{value}")
    return "\n\n".join(parts)

def _extract_interactions_with_llm(client: OpenAI, combined_text: str) -> Dict[str, List[str]]:
    instruction = (
        "You are extracting safety information from FDA drug labels. "
        "From the provided label excerpts, extract three arrays in strict JSON with keys: "
        "medications, diseases, do_not.\n\n"
        "- medications: other drugs, classes, or substances that should be avoided or require caution due to interactions (e.g., MAO inhibitors, warfarin, benzodiazepines, alcohol).\n"
        "- diseases: patient conditions where use is contraindicated or requires caution (e.g., liver disease, glaucoma, asthma, pregnancy).\n"
        "- do_not: concrete actions a patient should NOT do while using the drug (e.g., do not drink alcohol, do not drive, do not exceed 4000 mg/day).\n\n"
        "Return only a compact JSON object. No prose, no code fences."
    )

    response = client.responses.create(
        model="gpt-4.1",
        input=instruction + "\n\n=== FDA LABEL TEXT BEGIN ===\n" + combined_text + "\n=== FDA LABEL TEXT END ===",
        temperature=0,
        max_output_tokens=300,
    )
    text = response.output_text.strip()
    try:
        parsed = json.loads(text)
        medications = [m.strip() for m in parsed.get('medications', []) if isinstance(m, str) and m.strip()]
        diseases = [d.strip() for d in parsed.get('diseases', []) if isinstance(d, str) and d.strip()]
        do_not = [n.strip() for n in parsed.get('do_not', []) if isinstance(n, str) and n.strip()]
        return { 'medications': medications, 'diseases': diseases, 'do_not': do_not }
    except Exception:
        # Fallback: return empty structured result
        return { 'medications': [], 'diseases': [], 'do_not': [] }

def get_interactions_from_drug(drug: str) -> Dict[str, List[str]]:
    client = OpenAI()
    fetched = _fetch_openfda_labels(drug, limit=3)
    data = fetched.get('data', {})
    results = data.get('results', []) if isinstance(data, dict) else []
    if not results:
        return { 'medications': [], 'diseases': [], 'do_not': [] }

    # Combine the most recent labels' relevant sections
    combined_text_parts: List[str] = []
    for label in results:
        try:
            combined_text_parts.append(_combine_label_texts(label))
        except Exception:
            continue
    combined_text = "\n\n\n".join(p for p in combined_text_parts if p)

    extracted = _extract_interactions_with_llm(client, combined_text)

    # Persist a human-auditable record alongside the extraction
    try:
        record = {
            'drug': drug,
            'openfda_url': fetched.get('url'),
            'extracted': extracted,
            'fetched_at': fetched.get('fetched_at'),
        }
        # Append to a newline-delimited JSON log for traceability
        with open('response_interactions_openfda.json', 'a') as outfile:
            outfile.write(json.dumps(record) + "\n")
    except Exception:
        pass

    return extracted

def extract_drugs_from_conversation(conversation: Dict[str, Any]):
    client = OpenAI()
    try:
        messages = [
            {"role": "system", "content": "Answer with a comma separated list. e.g. acetaminophen,cetuximab or "},
            {"role": "user", "content": str(conversation['prompt'])},
            {"role": "developer", "content": f"Extract all of the medications relevant to this conversation. The medication might be mentioned explicitly in the conversation or relevant to the treatment or question. Return the generic names in a comma separated list."}
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
        )
        drugs = response.choices[0].message.content.strip()
        # Rate limit
        time.sleep(0.001)
        
    except Exception as e:
        print(f"Error processing conversation {conversation['prompt_id']}: {e}")
        drugs = []

    return drugs
    
if __name__ == "__main__":
    with open('datasets/healthbench_interactions.jsonl', 'r') as infile:
        conversations = [json.loads(line) for line in infile]
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    def process_conversation(conversation):
        drugs = extract_drugs_from_conversation(conversation)
        interactions = []
        for drug in drugs:
            interactions += get_interactions_from_drug(drug)
        conversation['interactions'] = interactions
        conversation['drugs'] = drugs
        return conversation

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_conversation, conversations), total=len(conversations), desc="Processing conversations"))

    # Update conversations in place (if needed later in the script)
    conversations = results

    # Save responses to JSON file
    print(f"Saving {len(conversations)} responses to response_interactions.json...")
    with open('response_interactions.json', 'w') as outfile:
        json.dump(conversations, outfile, indent=2)
    
    print(f"Successfully processed {len(conversations)} conversations")
