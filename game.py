from teams import BlueTeam, RedTeam
import pandas as pd
import json, os
import datetime
import random

# Configuration
RED_MODEL = "gpt-4"
BLUE_MODEL = "gpt-3.5-turbo"
RED_PROMPT_FILE = "prompts/red_example"
BLUE_PROMPT_FILE = "prompts/blue_example"

# Output logs
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Get script start timestamp
script_start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Initialize teams
red = RedTeam(RED_MODEL, RED_PROMPT_FILE)
blue = BlueTeam(BLUE_MODEL, BLUE_PROMPT_FILE)

# Load treatment questions from CSV
df = pd.read_csv("datasets/treatment.csv")
random_questions = df.sample(n=20, random_state=42)

for i, (_, row) in enumerate(random_questions.iterrows()):
    question = row['question']
    correct_answer = row['op' + 'abcd'[row['cop']]]

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
    with open(os.path.join(LOGS_DIR, f"scenario_{script_start_time}_{i:03d}.json"), "w") as f:
        json.dump(log_entry, f, indent=2)