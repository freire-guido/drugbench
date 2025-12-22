from openai import NotFoundError, OpenAI, APITimeoutError, APIConnectionError
import time, json, re
import fcntl

def safe_update_json(path, new_json):
    with open(path, 'r+') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            existing_json = json.load(f)
            existing_json.update(new_json)
            f.seek(0)
            json.dump(existing_json, f, indent=2)
            f.truncate()
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
            raise e
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def safe_update_jsonl(path, new_jsonl):
    with open(path, 'r+') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            existing_jsonl = [json.loads(line) for line in f if line.strip()]
            assert len(existing_jsonl) == len(new_jsonl), "Existing JSONL length does not match new JSONL length"
            for i in range(len(existing_jsonl)):
                existing_jsonl[i].update(new_jsonl[i])
            f.seek(0)
            for i in range(len(existing_jsonl)):
                f.write(json.dumps(existing_jsonl[i]) + '\n')
            f.truncate()
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
            raise e
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

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
        with open(batch_file, 'w') as f:
            for line in lines:
                f.write(json.dumps(line) + '\n')
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

def wait_for_batch(client, batch_id: str, delay: int = 5, timeout: int = 86400) -> bool:
    # 86400 seconds = 24 hours
    start_time = time.time()
    while client.batches.retrieve(batch_id).status in ["validating", "in_progress", "finalizing"] and (time.time() - start_time) < timeout:
        time.sleep(delay)
    if client.batches.retrieve(batch_id).status != "completed":
        raise Exception(f"Batch {batch_id} failed to complete: status {client.batches.retrieve(batch_id).status} after {time.time() - start_time} seconds")
    print(f"Batch {batch_id} completed in {time.time() - start_time} seconds")
    return True

def generate_batch_request(
    custom_id: str,
    messages: list[dict],
    model: str,
    top_logprobs: int = 0,
) -> list[dict]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "logprobs": top_logprobs > 0,
            "top_logprobs": top_logprobs if top_logprobs > 0 else None
        }
    }

def create_batch_job(
    client: OpenAI,
    requests: list[dict],
    input_file_name: str,
    verbose: bool = False
) -> str:
    if verbose:
        print(f"Creating batch job for {len(requests)} requests...")
    with open(input_file_name, 'w+') as outfile:
        for request in requests:
            outfile.write(json.dumps(request) + '\n')
    
    batch_input_file = client.files.create(file=open(input_file_name, "rb"), purpose="batch")
    batch_job = client.batches.create(input_file_id=batch_input_file.id, endpoint="/v1/chat/completions", completion_window="24h")
    return batch_job.id

def run_batch_with_tracker(
    client: OpenAI,
    requests: list[dict],
    input_file_name: str,
    tracker: dict | None = None,
    tracker_key: str | None = None,
    tracker_path: str | None = None,
    verbose: bool = False
) -> str:
    if tracker is not None and tracker_key in tracker:
        if verbose:
            print(f"Using batch ID from tracker for {tracker_key}")
        return tracker[tracker_key]
    else:
        batch_id = create_batch_job(client, requests, input_file_name, verbose)
        if tracker is not None:
            tracker[tracker_key] = batch_id
            if tracker_path is not None:
                safe_update_json(tracker_path, {tracker_key: batch_id})
                if verbose:
                    print(f"Saved batch ID {batch_id} to tracker ({tracker_key})")
        wait_for_batch(client, batch_id)
        return batch_id