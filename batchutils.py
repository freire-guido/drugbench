import time, json, re
from openai import NotFoundError

def parse_json_to_dict(json_string: str) -> dict:
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {}

def read_responses_batch(iterable) -> dict:
    res = [json.loads(line) if isinstance(line, str) else line for line in iterable]
    return {res['custom_id']: res['response']['body']['choices'][0]['message']['content'] for res in res}

def read_batch_output(client, batch_id: str, batch_file: str | None = None) -> list[dict]:
    try:
        file_id = client.batches.retrieve(batch_id).output_file_id
        file_content_response = client.files.content(file_id)
        text = file_content_response.content.decode('utf-8')
        lines = [json.loads(line) for line in text.split('\n') if line.strip()]
    except NotFoundError as e:
        print(f"Batch {batch_id} not found: {e} reading from {batch_file}...")
        with open(batch_file, 'r') as f:
            lines = [json.loads(line) for line in f]
    return lines

def sanitize_response(response: str, prescription: str | None = None) -> str:
    # replace <<prescription>> with the prescription from responses_red
    if prescription:
        response = response.replace('<<prescription>>', prescription)
    # remove the <explanation>
    response = re.sub(r'<explanation>.*?</explanation>', '', response, flags=re.DOTALL)
    return response

def wait_for_batch(client,batch_id: str, delay: int = 5, timeout: int = 86400) -> bool:
    # 86400 seconds = 24 hours
    start_time = time.time()
    while client.batches.retrieve(batch_id).status in ["validating", "in_progress", "finalizing"] and (time.time() - start_time) < timeout:
        time.sleep(delay)
    if client.batches.retrieve(batch_id).status != "completed":
        raise Exception(f"Batch {batch_id} failed to complete: status {client.batches.retrieve(batch_id).status} after {time.time() - start_time} seconds")
    print(f"Batch {batch_id} completed in {time.time() - start_time} seconds")
    return True