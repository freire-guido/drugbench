import json, time
import urllib.parse, urllib.request
from typing import Dict, Any, List

from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

OPENFDA_CACHE_PATH = 'datasets/openfda_cache.json'
OUT_PATH = 'datasets/drug_interactions.json'

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
    cache_key = f"{drug.lower()}|{limit}"
    if cache_key in cache:
        print(f"Using cached OpenFDA labels for {drug}")
        return cache[cache_key]

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
        return { 'url': url, 'data': [], 'fetched_at': time.time() }

    cache[cache_key] = { 'url': url, 'data': data, 'fetched_at': time.time() }
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
    parts = []
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
        "- do_not: concrete actions a patient should NOT do while using the drug (e.g., do not drink alcohol, do not drive, do not exceed 4000 mg/day). Only name the action e.g. if the label says 'do not drink alcohol', return 'drink alcohol'.\n\n"
        "Return only a compact JSON object. No prose, no code fences."
    )

    response = client.responses.create(
        model="gpt-5-nano",
        input=instruction + "\n\n=== FDA LABEL TEXT BEGIN ===\n" + combined_text + "\n=== FDA LABEL TEXT END ==="
    )

    text = response.output_text.strip()
    try:
        parsed = json.loads(text)
        medications = [m.strip() for m in parsed.get('medications', []) if isinstance(m, str) and m.strip()]
        diseases = [d.strip() for d in parsed.get('diseases', []) if isinstance(d, str) and d.strip()]
        do_not = [n.strip() for n in parsed.get('do_not', []) if isinstance(n, str) and n.strip()]
        return { 'medications': medications, 'diseases': diseases, 'do_not': do_not }
    except Exception as e:
        print(f"Error extracting interactions with LLM: {e}")
        return { 'medications': [], 'diseases': [], 'do_not': [] }

def get_interactions_from_drug(drug: str) -> Dict[str, List[str]]:
    print(drug)
    client = OpenAI()
    fetched = _fetch_openfda_labels(drug, limit=1)
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

    return extracted

def extract_drugs_from_conversation(conversation: Dict[str, Any]):
    client = OpenAI()
    try:
        messages = [
            {"role": "system", "content": "Answer with a comma separated list. e.g. acetaminophen,cetuximab"},
            {"role": "user", "content": str(conversation['prompt'])},
            {"role": "developer", "content": f"Extract up to four medications relevant to this conversation. The medication might be mentioned explicitly in the conversation or relevant to the treatment or question. Return the generic names in a comma separated list."}
        ]
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages
        )
        drugs = response.choices[0].message.content.strip()
        drugs = drugs.split(',')
        drugs = [drug.strip() for drug in drugs if drug.strip()]
        drugs = [drug for drug in drugs if drug not in ['', ' ', None, 'none', 'None']]
        drugs = drugs[:4]
        
    except Exception as e:
        print(f"Error processing conversation {conversation['prompt_id']}: {e}")
        drugs = []

    return drugs
    
if __name__ == "__main__":
    cache = _read_cache()
    drug_interactions = {}
    with open('datasets/healthbench_interactions.jsonl', 'r') as infile:
        conversations = [json.loads(line) for line in infile]

    def process_conversation(conversation):
        drugs = extract_drugs_from_conversation(conversation)
        for drug in drugs:
            if drug not in drug_interactions:
                drug_interactions[drug] = get_interactions_from_drug(drug)

    with ThreadPoolExecutor() as executor:
        tqdm(executor.map(process_conversation, conversations), total=len(conversations), desc="Processing conversations")

    # Save responses to JSON file
    print(f"Saving {len(drug_interactions)} drug interactions to {OUT_PATH}")
    _write_cache(cache)
    with open(OUT_PATH, 'w') as outfile:
        json.dump(drug_interactions, outfile, indent=2)
    
    print(f"Successfully processed {len(conversations)} conversations")
