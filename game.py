from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import os, json

from teams import BlueTeam, RedTeam
from eval import Evaluator

# Configuration
RED_MODEL = "gpt-5-nano"
BLUE_MODEL = "gpt-4o-mini"
EVALUATOR_MODEL = "gpt-4.1-nano"
RED_PROMPT_FILE = "prompts/red_long"
BLUE_PROMPT_FILE = "prompts/blue_chain"
LOGS_DIR = "logs"

# Initialize teams and evaluator
red = RedTeam(RED_MODEL, RED_PROMPT_FILE)
blue = BlueTeam(BLUE_MODEL, BLUE_PROMPT_FILE)
evaluator = Evaluator(EVALUATOR_MODEL)

# Load treatment questions from CSV
N_SCENARIOS = 100
df = pd.read_csv("datasets/treatment_curated.csv", index_col=0)
df['correct_answer'] = df.apply(lambda row: row['op'+'abcd'[row['cop']]], axis=1)

random_questions = df.sample(n=N_SCENARIOS, random_state=42)

def run_scenario(question, correct_answer):
    red_log, bias = red.answer(question)
    blue_log = blue.review_and_edit(question, red_log[-1])
    evaluator_log = evaluator.eval_question_answer(question, red_log[-1], correct_answer)
    return question, correct_answer, red_log, blue_log, evaluator_log, bias

start_time = datetime.now()

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = executor.map(run_scenario, list(random_questions['question']), list(random_questions['correct_answer']))
    for i, future in enumerate(tqdm(futures, total=N_SCENARIOS)):
        question, correct_answer, red_log, blue_log, evaluator_log, bias = future
        log_entry = {
            'question': question,
            'correct_answer': correct_answer,
            'red_log': red_log,
            'blue_log': blue_log,
            'evaluator_log': evaluator_log,
            'bias': bias,
        }
        with open(os.path.join(LOGS_DIR, f"scenario_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}_{i:03d}.json"), "w") as f:
            json.dump(log_entry, f, indent=2)