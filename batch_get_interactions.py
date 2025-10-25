import json, time, os
import urllib.parse, urllib.request
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

OPENFDA_CACHE_PATH = 'datasets/openfda_cache.json'
OUT_DRUGS_PATH = 'datasets/drug_interactions_hard.json'
BATCH_INPUT_PATH = 'batch/get_interactions_hard.jsonl'
BATCH_OUTPUT_PATH = 'batch/get_interactions_hard_output.jsonl'

def _read_cache(cache_path: str) -> Dict[str, Any]:
    """Read cache from file"""
    try:
        with open(cache_path, 'r') as infile:
            return json.load(infile)
    except Exception as e:
        print(f"Error reading cache {cache_path}: {e}")
        return {}

def _write_cache(cache: Dict[str, Any] | List[Any], cache_path: str) -> None:
    """Write cache to file"""
    try:
        with open(cache_path, 'w') as outfile:
            json.dump(cache, outfile)
        return True
    except Exception as e:
        print(f"Error writing cache {cache_path}: {e}")
        pass

def _fetch_openfda_labels(drug: str, limit: int = 3) -> Dict[str, Any]:
    """Fetch OpenFDA labels for a drug"""
    cache_key = f"{drug.lower()}|{limit}"
    if cache_key in fda_cache:
        print(f"Using cached OpenFDA labels for {drug}")
        return fda_cache[cache_key]

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

    fda_cache[cache_key] = { 'url': url, 'data': data, 'fetched_at': time.time() }
    return fda_cache[cache_key]

def _combine_label_texts(label: Dict[str, Any]) -> str:
    """Combine relevant sections from FDA label"""
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

def extract_drugs_from_batch_output(batch_output_path: str) -> List[str]:
    """Extract unique drugs from batch output file"""
    unique_drugs = set()
    
    try:
        with open(batch_output_path, 'r') as infile:
            for line in infile:
                result = json.loads(line)
                if result.get('error') is None and result.get('response', {}).get('status_code') == 200:
                    content = result.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
                    if content and content.strip():
                        # Parse comma-separated drug names
                        drugs = [drug.strip() for drug in content.split(',') if drug.strip()]
                        for drug in drugs:
                            if drug.lower() not in ['none', 'n/a', 'na', '']:
                                unique_drugs.add(drug.strip())
    except Exception as e:
        print(f"Error reading batch output: {e}")
        return []
    
    return list(unique_drugs)

def process_drug_for_batch(drug: str) -> Dict[str, Any]:
    """Process a single drug to create batch request data"""
    # First, fetch OpenFDA data to get label text
    fda_data = _fetch_openfda_labels(drug, limit=1)
    data = fda_data.get('data', {})
    results = data.get('results', []) if isinstance(data, dict) else []
    
    if not results:
        return None
        
    # Combine label texts
    combined_text_parts = []
    for label in results:
        try:
            combined_text_parts.append(_combine_label_texts(label))
        except Exception:
            continue
    combined_text = "\n\n\n".join(p for p in combined_text_parts if p)
    
    if not combined_text:
        return None
        
    # Create batch request for LLM extraction
    instruction = (
        "You are extracting safety information from FDA drug labels. "
        "From the provided label excerpts, extract three arrays in strict JSON with keys: "
        "medications, diseases, do_not.\n\n"
        "- medications: other drugs, classes, or substances that should be avoided or require caution due to interactions (e.g., MAO inhibitors, warfarin, benzodiazepines, alcohol).\n"
        "- diseases: patient conditions where use is contraindicated or requires caution (e.g., liver disease, glaucoma, asthma, pregnancy).\n"
        "- do_not: concrete actions a patient should NOT do while using the drug (e.g., do not drink alcohol, do not drive, do not exceed 4000 mg/day). Only name the action e.g. if the label says 'do not drink alcohol', return 'drink alcohol'.\n\n"
        "For each of the three arrays, return up to four of the most harmful interactions, diseases, or do_nots."
        "Return only a compact JSON object. No prose, no code fences."
    )
    
    request = {
        "custom_id": f"interactions_{drug}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-5-nano",
            "messages": [
                {
                    "role": "user", 
                    "content": instruction + "\n\n=== FDA LABEL TEXT BEGIN ===\n" + combined_text + "\n=== FDA LABEL TEXT END ==="
                }
            ],
        }
    }
    return request

def create_batch_requests(drugs: List[str]) -> None:
    """Create batch requests for drug interaction extraction using parallel processing"""
    print(f"Processing {len(drugs)} drugs with parallel OpenFDA calls...")
    
    # Use ThreadPoolExecutor for parallel OpenFDA calls
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_drug_for_batch, drugs), 
            total=len(drugs), 
            desc="Fetching OpenFDA data"
        ))
    
    # Filter out None results and create requests
    requests = [req for req in results if req is not None]
    
    # Write batch requests to file
    with open(BATCH_INPUT_PATH, 'w') as outfile:
        for request in requests:
            outfile.write(json.dumps(request) + '\n')
    
    print(f"Created {len(requests)} batch requests for drug interactions")

def process_batch_results(batch_output_path: str) -> Dict[str, Dict[str, List[str]]]:
    """Process batch results and extract drug interactions"""
    drug_interactions = {}
    
    try:
        with open(batch_output_path, 'r') as infile:
            for line in infile:
                result = json.loads(line)
                if result.get('error') is None and result.get('response', {}).get('status_code') == 200:
                    custom_id = result.get('custom_id', '')
                    if custom_id.startswith('interactions_'):
                        drug = custom_id.replace('interactions_', '')
                        content = result.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                        try:
                            parsed = json.loads(content)
                            medications = [m.strip() for m in parsed.get('medications', []) if isinstance(m, str) and m.strip()]
                            diseases = [d.strip() for d in parsed.get('diseases', []) if isinstance(d, str) and d.strip()]
                            do_not = [n.strip() for n in parsed.get('do_not', []) if isinstance(n, str) and n.strip()]
                            drug_interactions[drug] = {
                                'medications': medications,
                                'diseases': diseases,
                                'do_not': do_not
                            }
                        except Exception as e:
                            print(f"Error parsing interactions for {drug}: {e}")
                            drug_interactions[drug] = {
                                'medications': [],
                                'diseases': [],
                                'do_not': []
                            }
    except Exception as e:
        print(f"Error processing batch results: {e}")
        return {}
    
    return drug_interactions

def main():
    """Main function to orchestrate the batch drug interaction extraction"""
    global fda_cache
    fda_cache = _read_cache(OPENFDA_CACHE_PATH)
    
    # Check if we have batch output from drug extraction
    batch_output_path = input("Enter path to batch output file from drug extraction (or press Enter to skip): ").strip()
    
    if not batch_output_path:
        print("No batch output file provided. Exiting.")
        return
    
    # Extract unique drugs from batch output
    print("Extracting unique drugs from batch output...")
    drugs = extract_drugs_from_batch_output(batch_output_path)
    print(f"Found {len(drugs)} unique drugs")
    
    if not drugs:
        print("No drugs found in batch output. Exiting.")
        return
    
    # Create batch requests
    print("Creating batch requests for drug interactions...")
    create_batch_requests(drugs)
    
    # Upload batch file and create batch job
    client = OpenAI()
    
    print("Uploading batch file...")
    batch_input_file = client.files.create(
        file=open(BATCH_INPUT_PATH, "rb"),
        purpose="batch"
    )
    print(f"Batch input file ID: {batch_input_file.id}")
    
    print("Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch job ID: {batch_job.id}")
    print(f"Batch job status: {batch_job.status}")
    
    # Save cache
    _write_cache(fda_cache, OPENFDA_CACHE_PATH)
    
    print("\n" + "="*60)
    print("BATCH JOB CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"Batch Job ID: {batch_job.id}")
    print(f"Number of requests: {len(drugs)}")
    print(f"Estimated completion: 24 hours")
    print("\nTo process results when complete, run:")
    print(f"python batch_get_interactions.py process {batch_job.id}")
    print("\nTo check status manually:")
    print(f"client.batches.retrieve('{batch_job.id}')")
    print("="*60)

def process_completed_batch(batch_job_id: str):
    """Process a completed batch job and save results"""
    client = OpenAI()
    
    try:
        batch_job = client.batches.retrieve(batch_job_id)
        if batch_job.status != "completed":
            print(f"Batch job status is {batch_job.status}, not completed yet.")
            return
        
        # Check if we already have the output file locally
        local_output_path = f"batch/{batch_job_id}_output.jsonl"
        if os.path.exists(local_output_path):
            print(f"Using existing local file: {local_output_path}")
            batch_output_path = local_output_path
        else:
            print("Downloading batch results...")
            output_file = client.files.retrieve(batch_job.output_file_id)
            
            # Try to get the file content
            try:
                # Method 1: Try to get content directly
                file_content = output_file.content
                content = file_content.decode('utf-8')
            except AttributeError:
                try:
                    # Method 2: Try to read the file content
                    content = output_file.read()
                except AttributeError:
                    # Method 3: Use the file object directly
                    content = str(output_file)
            
            # Save to local file
            with open(BATCH_OUTPUT_PATH, 'w') as f:
                f.write(content)
            batch_output_path = BATCH_OUTPUT_PATH
        
        print("Processing batch results...")
        drug_interactions = process_batch_results(batch_output_path)
        
        # Load existing drug interactions
        existing_interactions = _read_cache(OUT_DRUGS_PATH)
        existing_interactions.update(drug_interactions)
        
        # Save updated drug interactions
        _write_cache(existing_interactions, OUT_DRUGS_PATH)
        print(f"Saved {len(drug_interactions)} drug interactions to {OUT_DRUGS_PATH}")
        
    except Exception as e:
        print(f"Error processing completed batch: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "process":
        if len(sys.argv) > 2:
            batch_job_id = sys.argv[2]
            process_completed_batch(batch_job_id)
        else:
            print("Usage: python batch_get_interactions.py process <batch_job_id>")
    else:
        main()
