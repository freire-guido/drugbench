import time, json

def read_batch_output(client, batch_id: str):
    batch_output = client.files.retrieve(client.batches.retrieve(batch_id).output_file_id)
    batch_output = batch_output.read().decode('utf-8')
    batch_output = [json.loads(line) for line in batch_output.split('\n') if line.strip()]
    return batch_output

def wait_for_batch(client,batch_id: str, delay: int = 5, timeout: int = 86400):
    # 86400 seconds = 24 hours
    start_time = time.time()
    while client.batches.retrieve(batch_id).status in ["validating", "in progress", "finalizing"] and (time.time() - start_time).total_seconds() < timeout:
        time.sleep(delay)
    if client.batches.retrieve(batch_id).status != "completed":
        raise Exception(f"Batch {batch_id} failed to complete: status {client.batches.retrieve(batch_id).status} after {time.time() - start_time} seconds")