from openai import NotFoundError, OpenAI, APITimeoutError, APIConnectionError
import time, json, re
import fcntl
import os
from dotenv import load_dotenv

load_dotenv()

# Import providers conditionally
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai
except ImportError:
    genai = None

def detect_provider(model_name: str) -> str:
    """Detect provider from model name prefix.
    
    Args:
        model_name: Model name (e.g., 'gpt-5', 'claude-opus-4-20250514', 'gemini-3-pro')
    
    Returns:
        'openai', 'anthropic', or 'gemini'
    """
    model_lower = model_name.lower()
    if model_lower.startswith('claude-') or model_lower.startswith('opus-') or model_lower.startswith('sonnet-'):
        return 'anthropic'
    elif model_lower.startswith('gemini-'):
        return 'gemini'
    else:
        return 'openai'

def get_client(provider: str):
    """Get client instance for the specified provider.
    
    Args:
        provider: 'openai', 'anthropic', or 'gemini'
    
    Returns:
        Client instance for the provider
    """
    if provider == 'openai':
        return OpenAI()
    elif provider == 'anthropic':
        if anthropic is None:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        return anthropic.Anthropic(api_key=api_key)
    elif provider == 'gemini':
        if genai is None:
            raise ImportError("google-genai package is required. Install with: pip install google-genai")
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required")
        return genai.Client(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")

def convert_messages_to_anthropic(messages: list[dict]) -> tuple[list[dict], str]:
    """Convert OpenAI-style messages to Anthropic format.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
    
    Returns:
        Tuple of (messages_list, system_message)
        - messages_list: List of messages without system messages
        - system_message: Combined system message text (empty string if none)
    """
    system_parts = []
    anthropic_messages = []
    
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'system':
            system_parts.append(content)
        elif role == 'user':
            anthropic_messages.append({
                'role': 'user',
                'content': content
            })
        elif role == 'assistant':
            anthropic_messages.append({
                'role': 'assistant',
                'content': content
            })
    
    system_message = '\n\n'.join(system_parts) if system_parts else ''
    return anthropic_messages, system_message

def convert_messages_to_gemini(messages: list[dict]) -> tuple[list[dict], str]:
    """Convert OpenAI-style messages to Gemini format.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
    
    Returns:
        Tuple of (contents_list, system_instruction)
        - contents_list: List of content dicts with 'parts' and 'role'
        - system_instruction: Combined system instruction text (empty string if none)
    """
    system_parts = []
    contents = []
    
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'system':
            system_parts.append(content)
        else:
            # Map OpenAI roles to Gemini roles
            gemini_role = 'user' if role == 'user' else 'model'
            contents.append({
                'role': gemini_role,
                'parts': [{'text': content}]
            })
    
    system_instruction = '\n\n'.join(system_parts) if system_parts else ''
    return contents, system_instruction

def generate_batch_request_multi(
    custom_id: str,
    messages: list[dict],
    model: str,
    provider: str | None = None,
    top_logprobs: int = 0,
) -> dict:
    """Generate batch request for the specified provider.
    
    Args:
        custom_id: Unique identifier for the request
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        provider: Provider name ('openai', 'anthropic', 'gemini') or None to auto-detect
        top_logprobs: Number of top logprobs (OpenAI only)
    
    Returns:
        Provider-specific batch request dict
    """
    if provider is None:
        provider = detect_provider(model)
    
    if provider == 'openai':
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
    elif provider == 'anthropic':
        anthropic_messages, system_message = convert_messages_to_anthropic(messages)
        params = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": 4096,  # Default, can be made configurable
        }
        if system_message:
            params["system"] = system_message
        return {
            "custom_id": custom_id,
            "params": params
        }
    elif provider == 'gemini':
        contents, system_instruction = convert_messages_to_gemini(messages)
        # Gemini batch format for file-based: {"key": "...", "request": GenerateContentRequest}
        # Each line in JSONL should have key and request fields
        request = {
            "contents": contents,
        }
        if system_instruction:
            request["system_instruction"] = {"parts": [{"text": system_instruction}]}
        # Return format: {"key": custom_id, "request": {...}}
        return {
            "key": custom_id,
            "request": request
        }
    else:
        raise ValueError(f"Unknown provider: {provider}")

def create_batch_job_multi(
    client,
    requests: list[dict],
    input_file_name: str,
    provider: str,
    model: str,
    verbose: bool = False
) -> str:
    """Create batch job for the specified provider.
    
    Args:
        client: Provider client instance
        requests: List of batch request dicts
        input_file_name: Path to write input JSONL file
        provider: Provider name ('openai', 'anthropic', 'gemini')
        model: Model name
        verbose: Whether to print progress
    
    Returns:
        Batch job ID
    """
    if verbose:
        print(f"Creating {provider} batch job for {len(requests)} requests...")
    
    # Write JSONL file
    with open(input_file_name, 'w+') as outfile:
        for request in requests:
            outfile.write(json.dumps(request) + '\n')
    
    if provider == 'openai':
        batch_input_file = client.files.create(file=open(input_file_name, "rb"), purpose="batch")
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch_job.id
    elif provider == 'anthropic':
        # Anthropic batch API: create batch with requests directly
        # Read requests from JSONL file and convert to Anthropic format
        with open(input_file_name, 'r') as f:
            requests_list = []
            for line in f:
                if line.strip():
                    req = json.loads(line)
                    # Convert our format to Anthropic batch format
                    # req has: {"custom_id": "...", "params": {...}}
                    requests_list.append({
                        "custom_id": req["custom_id"],
                        "params": req["params"]
                    })
        
        # Create batch using Anthropic's messages.batches.create
        batch_job = client.messages.batches.create(requests=requests_list)
        return batch_job.id
    elif provider == 'gemini':
        # Gemini batch API: use file-based to preserve custom_id keys
        # Inline requests don't preserve keys in responses, so we use file-based
        # Upload file and use file-based batch
        try:
            from google.genai import types
            uploaded_file = client.files.upload(
                file=input_file_name,
                config=types.UploadFileConfig(
                    display_name=f"batch_input_{int(time.time())}",
                    mime_type='application/jsonl'
                )
            )
        except (ImportError, AttributeError):
            uploaded_file = client.files.upload(
                file=input_file_name,
                config={'display_name': f"batch_input_{int(time.time())}", 'mime_type': 'application/jsonl'}
            )
        
        # Create batch job using src parameter with the file name string
        # SDK expects src to be a string starting with "files/" (e.g. "files/abc123")
        # See: https://ai.google.dev/gemini-api/docs/batch-api
        batch_job = client.batches.create(
            model=f"models/{model}",
            src=uploaded_file.name,  # Pass the file name string, not the file object
            config={
                'display_name': f"batch_{int(time.time())}",
            }
        )
        
        return batch_job.name  # Gemini uses .name instead of .id
    else:
        raise ValueError(f"Unknown provider: {provider}")

def _openai_find_batch(client, batch_id: str):
    """Find a batch by ID using list pagination (workaround for api.batch.read scope)."""
    for batch in client.batches.list():
        if batch.id == batch_id:
            return batch
    raise Exception(f"Batch {batch_id} not found in batches list")


def wait_for_batch_multi(
    client,
    batch_id: str,
    provider: str,
    delay: int = 5,
    timeout: int = 86400
) -> bool:
    """Wait for batch job to complete.
    
    Args:
        client: Provider client instance
        batch_id: Batch job ID
        provider: Provider name ('openai', 'anthropic', 'gemini')
        delay: Polling delay in seconds
        timeout: Maximum wait time in seconds
    
    Returns:
        True if batch completed successfully
    """
    start_time = time.time()
    
    if provider == 'openai':
        max_retries = 3
        retry_delay = 2
        while (time.time() - start_time) < timeout:
            try:
                batch = _openai_find_batch(client, batch_id)
                status = batch.status
                if status == "completed":
                    break
                elif status in ["failed", "expired", "cancelled"]:
                    raise Exception(f"Batch {batch_id} failed with status: {status}")
                elif status not in ["validating", "in_progress", "finalizing"]:
                    # Unknown status, wait a bit and check again
                    time.sleep(delay)
                    continue
                # Still processing
                time.sleep(delay)
            except (APIConnectionError, APITimeoutError) as e:
                # Retry on connection errors
                if max_retries > 0:
                    max_retries -= 1
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise Exception(f"Connection error while checking batch {batch_id}: {e}")
        
        # Final check
        try:
            final_status = _openai_find_batch(client, batch_id).status
            if final_status != "completed":
                raise Exception(f"Batch {batch_id} failed to complete: status {final_status} after {time.time() - start_time} seconds")
        except (APIConnectionError, APITimeoutError) as e:
            # If we can't check final status but we've waited long enough, assume it might be done
            # The read_batch_output will handle errors if batch isn't actually complete
            print(f"Warning: Could not verify final batch status due to connection error: {e}")
    elif provider == 'anthropic':
        # Anthropic batch processing_status: in_progress, ended
        # Check processing_ended_at to see if it's actually completed
        poll_count = 0
        while True:
            try:
                batch = client.messages.batches.retrieve(batch_id)
                processing_status = getattr(batch, 'processing_status', None)
                processing_ended_at = getattr(batch, 'processing_ended_at', None)
                
                # Log status every 10 polls (or first poll)
                if poll_count % 10 == 0:
                    print(f"[Anthropic] Batch {batch_id}: status={processing_status}, ended_at={processing_ended_at}")
                poll_count += 1
                
                # Batch is done when processing_status is "ended"
                # (processing_ended_at may be None in some cases, so we don't require it)
                if processing_status == "ended":
                    break
                elif (time.time() - start_time) >= timeout:
                    raise Exception(f"Batch {batch_id} timed out after {time.time() - start_time} seconds")
                time.sleep(delay)
            except Exception as e:
                print(f"[Anthropic] Error polling batch {batch_id}: {e}")
                raise
    elif provider == 'gemini':
        # Gemini batch status: JOB_STATE_PENDING, JOB_STATE_RUNNING, JOB_STATE_SUCCEEDED, JOB_STATE_FAILED
        max_retries = 3
        retry_delay = 2
        while (time.time() - start_time) < timeout:
            try:
                batch = client.batches.get(name=batch_id)
                state = batch.state
                if state == "JOB_STATE_SUCCEEDED":
                    break
                elif state == "JOB_STATE_FAILED":
                    error_msg = getattr(batch, 'error', 'Unknown error')
                    raise Exception(f"Batch {batch_id} failed: {error_msg}")
                elif state in ["JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
                    raise Exception(f"Batch {batch_id} was {state.lower()}")
                # Still processing
                time.sleep(delay)
            except Exception as e:
                # Retry on connection errors
                if max_retries > 0 and ("connection" in str(e).lower() or "timeout" in str(e).lower()):
                    max_retries -= 1
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                elif "failed" in str(e).lower() or "cancelled" in str(e).lower() or "expired" in str(e).lower():
                    raise  # Don't retry on these errors
                else:
                    raise Exception(f"Error checking Gemini batch {batch_id}: {e}")
        
        if (time.time() - start_time) >= timeout:
            raise Exception(f"Batch {batch_id} timed out after {time.time() - start_time} seconds")
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    print(f"Batch {batch_id} completed in {time.time() - start_time} seconds")
    return True

def read_batch_output_multi(
    client,
    batch_id: str,
    batch_file: str | None,
    provider: str
) -> tuple[list[dict], list[str] | None]:
    """Read batch output from provider.
    
    Args:
        client: Provider client instance
        batch_id: Batch job ID
        batch_file: Path to save/read batch output file
        provider: Provider name ('openai', 'anthropic', 'gemini')
    
    Returns:
        List of result dicts
    """
    custom_id_mapping = None
    if provider == 'gemini' and batch_file:
        # Try to load custom_id mapping
        mapping_file = batch_file.replace('_output.jsonl', '_input_custom_id_mapping.json')
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                custom_id_mapping = json.load(f)
    
    if provider == 'openai':
        try:
            file_id = _openai_find_batch(client, batch_id).output_file_id
            file_content_response = client.files.content(file_id)
            text = file_content_response.content.decode('utf-8')
            lines = [json.loads(line) for line in text.split('\n') if line.strip()]
            if batch_file:
                with open(batch_file, 'w') as f:
                    for line in lines:
                        f.write(json.dumps(line) + '\n')
        except NotFoundError as e:
            print(f"Batch {batch_id} not found: {e} reading from {batch_file}...")
            if batch_file and os.path.exists(batch_file):
                with open(batch_file, 'r') as f:
                    lines = [json.loads(line) for line in f]
            else:
                raise
        return (lines, None)
    elif provider == 'anthropic':
        # Check if we have a cached output file first
        if batch_file and os.path.exists(batch_file):
            try:
                with open(batch_file, 'r') as f:
                    lines = [json.loads(line) for line in f if line.strip()]
                if lines:  # File has content, use it
                    print(f"Reading Anthropic batch {batch_id} from cache: {batch_file}")
                    return (lines, None)
            except Exception as e:
                print(f"Warning: Could not read cache file {batch_file}: {e}")
        
        # Stream results from Anthropic API using the correct method
        # According to Anthropic docs: client.messages.batches.results(batch_id)
        results = []
        print(f"Streaming Anthropic batch results for {batch_id}...")
        for result in client.messages.batches.results(batch_id):
            # Format result to match expected structure
            result_dict = {
                "custom_id": getattr(result, 'custom_id', ''),
                "result": result.result.type if hasattr(result.result, 'type') else 'unknown'
            }
            
            if result_dict["result"] == "succeeded":
                # Extract content from result.result.message
                message = result.result.message
                response_dict = {"content": []}
                if hasattr(message, 'content'):
                    for content_block in message.content:
                        if isinstance(content_block, dict):
                            response_dict["content"].append(content_block)
                        else:
                            # Convert object to dict
                            block_dict = {"type": getattr(content_block, 'type', 'text')}
                            if hasattr(content_block, 'text'):
                                block_dict["text"] = content_block.text
                            response_dict["content"].append(block_dict)
                result_dict["response"] = response_dict
            elif result_dict["result"] == "errored":
                # Handle error
                error_obj = result.result.error if hasattr(result.result, 'error') else None
                if error_obj:
                    if isinstance(error_obj, dict):
                        result_dict["error"] = error_obj
                    else:
                        result_dict["error"] = {"message": str(error_obj)}
            results.append(result_dict)
        
        print(f"Retrieved {len(results)} results from Anthropic batch {batch_id}")
        
        if batch_file:
            with open(batch_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
        return (results, None)
    elif provider == 'gemini':
        batch = client.batches.get(name=batch_id)
        
        # Check if this is an inline batch (has inlineResponses) or file-based (has dest.file_name)
        if hasattr(batch, 'inline_responses') and batch.inline_responses:
            # Inline responses - convert to list of dicts
            results = []
            for inline_resp in batch.inline_responses:
                # Convert inline response to dict format
                resp_dict = {}
                if hasattr(inline_resp, 'response'):
                    resp_dict['response'] = inline_resp.response
                elif hasattr(inline_resp, 'candidates'):
                    resp_dict['response'] = {'candidates': inline_resp.candidates}
                if hasattr(inline_resp, 'error'):
                    resp_dict['error'] = inline_resp.error
                results.append(resp_dict)
            
            if batch_file:
                with open(batch_file, 'w') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
            return (results, None)
        elif batch.dest and hasattr(batch.dest, 'file_name') and batch.dest.file_name:
            # File-based output - download the file
            output_file_name = batch.dest.file_name
            try:
                file_content = client.files.download(file=output_file_name)
                # File content might be bytes or text, handle both
                if isinstance(file_content, bytes):
                    text = file_content.decode('utf-8')
                else:
                    text = str(file_content)
                lines = [json.loads(line) for line in text.split('\n') if line.strip()]
                if batch_file:
                    with open(batch_file, 'w') as f:
                        for line in lines:
                            f.write(json.dumps(line) + '\n')
                return (lines, custom_id_mapping)
            except Exception as e:
                # Fallback: try reading from batch_file if it exists
                if batch_file and os.path.exists(batch_file):
                    print(f"Error downloading file {output_file_name}: {e}, reading from {batch_file}...")
                    with open(batch_file, 'r') as f:
                        lines = [json.loads(line) for line in f]
                    return (lines, custom_id_mapping)
                else:
                    raise Exception(f"Failed to download batch output: {e}")
        else:
            raise Exception(f"Batch {batch_id} has no output (neither inline_responses nor dest.file_name)")
    else:
        raise ValueError(f"Unknown provider: {provider}")

def read_responses_batch_multi(iterable, provider: str, custom_id_mapping: list[str] | None = None) -> dict:
    """Extract responses from batch output.
    
    Args:
        iterable: Iterable of result dicts
        provider: Provider name ('openai', 'anthropic', 'gemini')
    
    Returns:
        Dict mapping custom_id to response content
    """
    res = [json.loads(line) if isinstance(line, str) else line for line in iterable]
    
    if provider == 'openai':
        return {item['custom_id']: item['response']['body']['choices'][0]['message']['content'] for item in res}
    elif provider == 'anthropic':
        result_dict = {}
        for item in res:
            custom_id = item.get('custom_id')
            result_status = item.get('result', '')
            
            if custom_id and result_status == 'succeeded' and 'response' in item:
                # Anthropic response format: response.content is a list of content blocks
                content_blocks = item['response'].get('content', [])
                # Extract text from content blocks
                text_parts = []
                for block in content_blocks:
                    # Handle both dict format and object format
                    if isinstance(block, dict):
                        if block.get('type') == 'text':
                            text_parts.append(block.get('text', ''))
                    else:
                        # Object format - try to get text attribute
                        if hasattr(block, 'type') and block.type == 'text':
                            text_parts.append(getattr(block, 'text', ''))
                result_dict[custom_id] = ''.join(text_parts)
            elif custom_id and result_status == 'errored':
                # Handle errors - log and set empty response
                error_info = item.get('error', {})
                error_msg = error_info.get('message', 'Unknown error') if isinstance(error_info, dict) else str(error_info)
                print(f"Warning: Error for {custom_id}: {error_msg}")
                result_dict[custom_id] = ""
        return result_dict
    elif provider == 'gemini':
        result_dict = {}
        # For Gemini, responses can be from inline (list of inlineResponse) or file-based (JSONL with key/response)
        # Handle both formats
        for item in res:
            # For inline responses, item might be a dict with response directly
            # For file-based, item has "key" and "response" or "error"
            key = None
            response_data = None
            error_data = None
            
            if 'key' in item:
                # File-based format
                key = item.get('key')
                if 'response' in item:
                    response_data = item['response']
                elif 'error' in item:
                    error_data = item['error']
            else:
                # Inline format - use index or try to extract from response
                # For inline, we need to match by order since there's no key
                # This is a limitation - we'll use index
                idx = res.index(item)
                key = f"request_{idx}"
                if 'response' in item or 'candidates' in item:
                    response_data = item.get('response', item)
                elif 'error' in item:
                    error_data = item.get('error')
            
            if not key:
                continue
            
            if response_data:
                # Extract text from response
                # Format: response.candidates[0].content.parts or inlineResponse format
                candidates = []
                if isinstance(response_data, dict):
                    candidates = response_data.get('candidates', [])
                elif hasattr(response_data, 'candidates'):
                    candidates = response_data.candidates
                
                if candidates:
                    # Get first candidate
                    candidate = candidates[0] if isinstance(candidates, list) else candidates
                    if isinstance(candidate, dict):
                        content = candidate.get('content', {})
                    else:
                        content = getattr(candidate, 'content', {})
                    
                    if isinstance(content, dict):
                        parts = content.get('parts', [])
                    else:
                        parts = getattr(content, 'parts', [])
                    
                    text_parts = []
                    for part in parts:
                        if isinstance(part, dict):
                            if 'text' in part:
                                text_parts.append(part['text'])
                        else:
                            if hasattr(part, 'text'):
                                text_parts.append(part.text)
                    result_dict[key] = ''.join(text_parts)
            elif error_data:
                # Handle error responses
                if isinstance(error_data, dict):
                    error_msg = error_data.get('message', 'Unknown error')
                else:
                    error_msg = str(error_data)
                print(f"Warning: Error for {key}: {error_msg}")
                result_dict[key] = ""  # Empty response for errors
        
        return result_dict
    else:
        raise ValueError(f"Unknown provider: {provider}")

def run_batch_with_tracker_multi(
    client,
    requests: list[dict],
    input_file_name: str,
    tracker: dict | None,
    tracker_key: str | None,
    tracker_path: str | None,
    provider: str,
    model: str,
    verbose: bool = False
) -> str:
    """Run batch job with tracker support (multi-provider).
    
    Args:
        client: Provider client instance
        requests: List of batch request dicts
        input_file_name: Path to write input JSONL file
        tracker: Tracker dict
        tracker_key: Key to store batch ID in tracker
        tracker_path: Path to tracker JSON file
        provider: Provider name ('openai', 'anthropic', 'gemini')
        model: Model name
        verbose: Whether to print progress
    
    Returns:
        Batch job ID
    """
    from batchutils import safe_update_json
    
    if tracker is not None and tracker_key in tracker:
        batch_id = tracker[tracker_key]
        if verbose:
            print(f"Using batch ID from tracker for {tracker_key}: {batch_id}")
        # Check if batch is already complete, if not wait for it
        # This handles cases where batch was created but script was interrupted
        try:
            if provider == 'openai':
                try:
                    status = _openai_find_batch(client, batch_id).status
                    if status not in ["completed", "failed", "expired", "cancelled"]:
                        if verbose:
                            print(f"Batch {batch_id} still processing (status: {status}), waiting...")
                        wait_for_batch_multi(client, batch_id, provider)
                    elif verbose:
                        print(f"Batch {batch_id} already completed (status: {status})")
                except (APIConnectionError, APITimeoutError) as e:
                    # If we can't check status, assume it might be complete and let read_batch_output handle it
                    if verbose:
                        print(f"Warning: Could not check batch status, assuming complete: {e}")
            elif provider == 'anthropic':
                batch = client.messages.batches.retrieve(batch_id)
                processing_status = getattr(batch, 'processing_status', None)
                processing_ended_at = getattr(batch, 'processing_ended_at', None)
                if verbose:
                    print(f"Batch {batch_id} status: processing_status={processing_status}, ended_at={processing_ended_at}")
                if processing_status != "ended":
                    if verbose:
                        print(f"Batch {batch_id} still processing, waiting...")
                    wait_for_batch_multi(client, batch_id, provider)
                elif verbose:
                    print(f"Batch {batch_id} already completed")
            elif provider == 'gemini':
                batch = client.batches.get(name=batch_id)
                state = batch.state
                if state not in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
                    if verbose:
                        print(f"Batch {batch_id} still processing, waiting...")
                    wait_for_batch_multi(client, batch_id, provider)
        except Exception as e:
            # If we can't check status, assume it needs to be created
            if verbose:
                print(f"Could not check batch status: {e}, creating new batch...")
            batch_id = create_batch_job_multi(client, requests, input_file_name, provider, model, verbose)
            if tracker is not None:
                tracker[tracker_key] = batch_id
                if tracker_path is not None:
                    safe_update_json(tracker_path, {tracker_key: batch_id})
                    if verbose:
                        print(f"Saved batch ID {batch_id} to tracker ({tracker_key})")
            wait_for_batch_multi(client, batch_id, provider)
        return batch_id
    else:
        batch_id = create_batch_job_multi(client, requests, input_file_name, provider, model, verbose)
        if tracker is not None:
            tracker[tracker_key] = batch_id
            if tracker_path is not None:
                safe_update_json(tracker_path, {tracker_key: batch_id})
                if verbose:
                    print(f"Saved batch ID {batch_id} to tracker ({tracker_key})")
        wait_for_batch_multi(client, batch_id, provider)
        return batch_id

# Re-export utility functions from batchutils
from batchutils import safe_update_json, safe_update_jsonl, parse_json_to_dict, sanitize_response

