from eval import Evaluator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import time
import random
from openai import RateLimitError

load_dotenv()

client = OpenAI()
evaluator = Evaluator("gpt-4")

df = pd.read_csv('datasets/treatment_curated.csv')
random_questions = df.sample(n=1000, random_state=42)

def retry_with_backoff(func, max_retries=5, base_delay=1.0):
    """Retry function with exponential backoff for rate limits"""
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            
            # Extract wait time from error message if available
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            if "Please try again in" in str(e):
                try:
                    # Try to extract the suggested wait time
                    error_msg = str(e)
                    if "ms" in error_msg:
                        suggested_wait = float(error_msg.split("Please try again in ")[1].split("ms")[0]) / 1000
                    elif "s" in error_msg:
                        suggested_wait = float(error_msg.split("Please try again in ")[1].split("s")[0])
                    else:
                        suggested_wait = wait_time
                    wait_time = max(wait_time, suggested_wait + random.uniform(0.5, 1.5))
                except:
                    pass
            
            print(f"Rate limit hit, waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)
        except Exception as e:
            # For non-rate-limit errors, don't retry
            raise e

def process_question(row_data):
    """Process a single question and return results"""
    idx, row = row_data
    question = row['question']
    question_id = row['id']
    correct_answer = row['op' + 'abcd'[row['cop']]]
    
    try:
        # Get answer with retry logic
        def get_answer():
            response = client.responses.create(
                model="gpt-5",
                input=[
                    {"role": "system", "content": "You are a assistant that answers medical questions in a short one or two sentences."},
                    {"role": "user", "content": question}
                ]
            )
            return response.output_text
        
        answer = retry_with_backoff(get_answer)
        
        # Get score with retry logic
        def get_score():
            return evaluator.eval_question_answer(question, answer, correct_answer)
        
        score = retry_with_backoff(get_score)
        
        return idx, question_id, question, answer, score
    except Exception as e:
        print(f"Error processing question {idx} (ID: {question_id}): {e}")
        return idx, question_id, question, "Error", "Error"

# Prepare data for parallel processing
question_data = list(random_questions.iterrows())

# Use ThreadPoolExecutor for parallel processing
results = {}
max_workers = 10  # Adjust based on your API rate limits

print(f"Processing {len(question_data)} questions with {max_workers} parallel workers...")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_idx = {executor.submit(process_question, data): data[0] for data in question_data}
    
    # Process completed tasks with progress bar
    for future in tqdm(as_completed(future_to_idx), total=len(question_data), desc="Processing"):
        idx, question_id, question, answer, score = future.result()
        results[idx] = {'id': question_id, 'question': question, 'answer': answer, 'score': score}

# Sort results by original index to maintain order
sorted_results = [results[idx] for idx in sorted(results.keys())]

# Create DataFrame
question_ids = [r['id'] for r in sorted_results]
questions = [r['question'] for r in sorted_results]
answers = [r['answer'] for r in sorted_results]
scores = [r['score'] for r in sorted_results]

pd.DataFrame({
    'id': question_ids,
    'question': questions, 
    'answer': answers, 
    'score': scores
}).to_csv('datasets/baselines_results_gpt5.csv', index=False)
print(f"Results saved to datasets/baselines_results_gpt5.csv with {len(sorted_results)} rows")