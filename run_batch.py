from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

batch_input_file = client.files.create(
    file=open("batch/answerable.jsonl", "rb"),
    purpose="batch"
)

print(batch_input_file)

batch_input_file_id = batch_input_file.id
res = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

print(res)