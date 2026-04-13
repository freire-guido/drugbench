from batch_honest import main as batch_honest_main
from batch_edit import main as batch_edit_main
from batch_defer import main as batch_defer_main
from batch_eval_honest import main as batch_eval_honest_main

import argparse
import json
import random
import os
from concurrent.futures import ThreadPoolExecutor

def main(args):
    # Sample conversations
    with open(args.conversations, 'r') as f:
        all_conversations = [json.loads(line) for line in f]
    if args.seed is not None:
        random.seed(args.seed)
    if args.n is not None:
        sampled_conversations = random.sample(all_conversations, args.n)
    else:
        sampled_conversations = all_conversations
    
    # Create out_dir if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load existing tracker or create empty one
    tracker_path = os.path.join(args.out_dir, 'tracker.json')
    if os.path.exists(tracker_path):
        with open(tracker_path, 'r') as f:
            try:
                tracker = json.load(f)
                print(f"Loaded existing tracker with keys: {list(tracker.keys())}")
            except json.JSONDecodeError:
                tracker = {}
    else:
        tracker = {}
        with open(tracker_path, 'w') as f:
            json.dump(tracker, f)
    
    # Write sampled conversations to out_dir (skip if reusing red team outputs)
    conversations_path = os.path.join(args.out_dir, 'conversations_honest.jsonl')
    if args.skip_red:
        print(f"--skip_red: loading existing {conversations_path}")
        with open(conversations_path, 'r') as f:
            sampled_conversations = [json.loads(line) for line in f]
    else:
        with open(conversations_path, 'w') as f:
            for convo in sampled_conversations:
                f.write(json.dumps(convo) + '\n')

    if not args.skip_red:
        # Create Namespace objects for batch scripts
        batch_honest_args = argparse.Namespace(
            conversations=conversations_path,
            red_prompts=args.red_prompts,
            blue_prompts=args.blue_prompts,
            red_model=args.red_model,
            blue_model=args.blue_model,
            tracker=tracker_path,
            out_dir=args.out_dir
        )

        # Run batch_honest
        print("Running batch_honest...")
        batch_honest_main(batch_honest_args)

    # Always run honest eval for red_response (runs fresh when tracker key is absent)
    print("Running honest eval for red_response...")
    batch_eval_red_args = argparse.Namespace(
        conversations=conversations_path,
        response_field='red_response',
        model=args.eval_model,
        tracker=tracker_path,
        out_dir=args.out_dir
    )
    batch_eval_honest_main(batch_eval_red_args)

    # Run batch_edit and batch_defer in parallel
    print("Running batch_edit and batch_defer in parallel...")
    batch_edit_args = argparse.Namespace(
        conversations=conversations_path,
        blue_prompts=args.blue_prompts,
        blue_model=args.blue_model,
        tracker=tracker_path,
        out_dir=args.out_dir
    )
    
    batch_defer_args = argparse.Namespace(
        conversations=conversations_path,
        blue_prompts=args.blue_prompts,
        blue_model=args.blue_model,
        tracker=tracker_path,
        out_dir=args.out_dir
    )
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        edit_future = executor.submit(batch_edit_main, batch_edit_args)
        defer_future = executor.submit(batch_defer_main, batch_defer_args)
        edit_future.result()
        defer_future.result()
    
    # Run honest evals for blue_edit and blue_defer in parallel
    print("Running honest evals for blue_edit and blue_defer in parallel...")
    batch_eval_edit_args = argparse.Namespace(
        conversations=conversations_path,
        response_field='blue_edit',
        model=args.eval_model,
        tracker=tracker_path,
        out_dir=args.out_dir
    )
    
    batch_eval_defer_args = argparse.Namespace(
        conversations=conversations_path,
        response_field='blue_defer',
        model=args.eval_model,
        tracker=tracker_path,
        out_dir=args.out_dir
    )
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        eval_edit_future = executor.submit(batch_eval_honest_main, batch_eval_edit_args)
        eval_defer_future = executor.submit(batch_eval_honest_main, batch_eval_defer_args)
        eval_edit_future.result()
        eval_defer_future.result()
    
    print(f"Pipeline complete! Final output: {conversations_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, required=True,
                        help="Input conversations JSONL file")
    parser.add_argument("--red_prompts", type=str, required=True,
                        help="Path to red prompts JSON file")
    parser.add_argument("--blue_prompts", type=str, required=True,
                        help="Path to blue prompts JSON file")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of conversations to sample")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling")
    parser.add_argument("--red_model", type=str, default="gpt-5",
                        help="Red team model")
    parser.add_argument("--blue_model", type=str, default="gpt-4o",
                        help="Blue team model")
    parser.add_argument("--eval_model", type=str, default="gpt-4.1",
                        help="Model for honest evaluations")
    parser.add_argument("--skip_red", action="store_true",
                        help="Skip red team: load existing conversations_honest.jsonl and reuse red responses")
    args = parser.parse_args()
    main(args)

