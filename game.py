from teams import BlueTeam, RedTeam
from eval import Evaluator
import pandas as pd
import json, os
import datetime
import random

# Configuration
RED_MODEL = "gpt-4"
BLUE_MODEL = "gpt-3.5-turbo"
EVALUATOR_MODEL = "gpt-4"
RED_PROMPT_FILE = "prompts/red_chain"
BLUE_PROMPT_FILE = "prompts/blue_chain"

# Output logs
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize teams and evaluator
red = RedTeam(RED_MODEL, RED_PROMPT_FILE)
blue = BlueTeam(BLUE_MODEL, BLUE_PROMPT_FILE)
evaluator = Evaluator(EVALUATOR_MODEL)

# Load treatment questions from CSV
N_SCENARIOS = 5
df = pd.read_csv("datasets/treatment.csv")
random_questions = df.sample(n=N_SCENARIOS, random_state=42)

loop_start_time = datetime.datetime.now()
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
        "evaluator_log": evaluator.eval_question_answer(question, red_log[-1], correct_answer),
        "bias": bias,
    }
    with open(os.path.join(LOGS_DIR, f"scenario_{loop_start_time.strftime('%Y-%m-%d_%H-%M-%S')}_{i:03d}.json"), "w") as f:
        json.dump(log_entry, f, indent=2)

    print("\033[F\033[K" * 3, end="")  # Move cursor up 3 lines and clear them
    print(f"{i+1}/{N_SCENARIOS}")
    print(f"Time: {datetime.datetime.now() - loop_start_time}")
    print(f"ETA: {(datetime.datetime.now() - loop_start_time) / (i+1) * (N_SCENARIOS - i - 1)}")