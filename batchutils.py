import time, json, re

def read_responses_batch(iterable) -> dict:
    res = [json.loads(line) if isinstance(line, str) else line for line in iterable]
    return {res['custom_id']: res['response']['body']['choices'][0]['message']['content'] for res in res}

def read_batch_output(client, batch_id: str) -> list[dict]:
    file_id = client.batches.retrieve(batch_id).output_file_id
    file_content_response = client.files.content(file_id)
    text = file_content_response.content.decode('utf-8')
    lines = [json.loads(line) for line in text.split('\n') if line.strip()]
    return lines

def sanitize_response(response: str, prescription: str | None = None) -> str:
    # replace <<prescription>> with the prescription from responses_red
    if prescription:
        response = response.replace('<<prescription>>', prescription)
    # remove the <explanation>
    response = re.sub(r'<explanation>.*?</explanation>', '', response)
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