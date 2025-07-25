from teams import BlueTeam, RedTeam
from datasets import load_dataset
import json, os
import datetime

# Configuration
RED_MODEL = "gpt-4"
BLUE_MODEL = "gpt-3.5-turbo"
RED_PROMPT_FILE = "prompts/red_example"
BLUE_PROMPT_FILE = "prompts/blue_example"

# Output logs
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize teams
red = RedTeam(RED_MODEL, RED_PROMPT_FILE)
blue = BlueTeam(BLUE_MODEL, BLUE_PROMPT_FILE)

dataset = load_dataset("openlifescienceai/medmcqa", split="train[:10]")  # small sample for demo

for i, item in enumerate(dataset):
    print(item)
    question = item['question']
    correct_answer = item['op' + 'abcd'[item['cop']]]

    red_log, bias = red.answer(question)
    blue_log = blue.review_and_edit(question, red_log[-1])

    log_entry = {
        "scenario_id": i,
        "question": question,
        "correct_answer": correct_answer,
        "red_log": red_log,
        "blue_log": blue_log,
        "bias": bias,
    }
    with open(os.path.join(LOGS_DIR, f"scenario_{datetime.datetime.today().strftime('%Y-%m-%d')}_{i}.json"), "w") as f:
        json.dump(log_entry, f, indent=2)