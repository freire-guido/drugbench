#!/usr/bin/env python3
"""
Run both attack and honest evaluation pipelines in parallel.

This script orchestrates the full experiment by running:
- run_attack.py: Evaluates model robustness against adversarial attacks
- run_honest.py: Evaluates model performance on honest (non-adversarial) inputs

Both pipelines run in parallel with progress tracking and shared configuration.
"""

from run_attack import main as run_attack_main
from run_honest import main as run_honest_main

import argparse
import os
import sys
import time
import threading
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


class ProgressTracker:
    """Thread-safe progress tracker with timestamps."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.events = []
        
    def log(self, pipeline: str, message: str):
        """Log a progress event with timestamp."""
        with self.lock:
            elapsed = time.time() - self.start_time
            timestamp = datetime.now().strftime("%H:%M:%S")
            event = f"[{timestamp}] [{elapsed:7.1f}s] [{pipeline:6s}] {message}"
            self.events.append(event)
            print(event, flush=True)
    
    def summary(self):
        """Print experiment summary."""
        total_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print(f"EXPERIMENT COMPLETE")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print("=" * 60)


def run_attack_with_progress(args, tracker: ProgressTracker):
    """Run attack pipeline with progress logging."""
    tracker.log("ATTACK", "Starting attack pipeline...")
    try:
        run_attack_main(args)
        tracker.log("ATTACK", "✓ Attack pipeline complete")
        return True, None
    except Exception as e:
        tb = traceback.format_exc()
        tracker.log("ATTACK", f"✗ Attack pipeline failed: {type(e).__name__}: {e}")
        print(f"\n{'='*60}\nATTACK PIPELINE TRACEBACK:\n{'='*60}\n{tb}", flush=True)
        return False, tb


def run_honest_with_progress(args, tracker: ProgressTracker):
    """Run honest pipeline with progress logging."""
    tracker.log("HONEST", "Starting honest pipeline...")
    try:
        run_honest_main(args)
        tracker.log("HONEST", "✓ Honest pipeline complete")
        return True, None
    except Exception as e:
        tb = traceback.format_exc()
        tracker.log("HONEST", f"✗ Honest pipeline failed: {type(e).__name__}: {e}")
        print(f"\n{'='*60}\nHONEST PIPELINE TRACEBACK:\n{'='*60}\n{tb}", flush=True)
        return False, tb


def main(args):
    # Generate output directory with timestamp if not specified
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"datasets/experiment_{timestamp}"
    
    # Create separate output directories for attack and honest pipelines
    attack_out_dir = os.path.join(args.out_dir, "attack")
    honest_out_dir = os.path.join(args.out_dir, "honest")
    os.makedirs(attack_out_dir, exist_ok=True)
    os.makedirs(honest_out_dir, exist_ok=True)
    
    # Print experiment configuration
    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"  Conversations:  {args.conversations}")
    print(f"  Sample size:    {args.n if args.n else 'all'}")
    print(f"  Seed:           {args.seed if args.seed else 'random'}")
    print(f"  Red model:      {args.red_model}")
    print(f"  Blue model:     {args.blue_model}")
    print(f"  Eval model:     {args.eval_model}")
    print(f"  Red prompts:    {args.red_prompts}")
    print(f"  Blue prompts:   {args.blue_prompts}")
    print(f"  Honest prompts: {args.honest_prompts}")
    print(f"  Output dir:     {args.out_dir}")
    print("=" * 60 + "\n")
    
    tracker = ProgressTracker()
    
    # Create argument namespaces for each pipeline
    attack_args = argparse.Namespace(
        conversations=args.conversations,
        red_prompts=args.red_prompts,
        blue_prompts=args.blue_prompts,
        out_dir=attack_out_dir,
        n=args.n,
        seed=args.seed,
        red_model=args.red_model,
        blue_model=args.blue_model,
        eval_model=args.eval_model
    )
    
    honest_args = argparse.Namespace(
        conversations=args.conversations,
        red_prompts=args.honest_prompts,  # Use honest prompts for "red" in honest pipeline
        blue_prompts=args.blue_prompts,
        out_dir=honest_out_dir,
        n=args.n,
        seed=args.seed,
        red_model=args.red_model,
        blue_model=args.blue_model,
        eval_model=args.eval_model
    )
    
    # Run both pipelines in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_attack_with_progress, attack_args, tracker): "attack",
            executor.submit(run_honest_with_progress, honest_args, tracker): "honest"
        }
        
        for future in as_completed(futures):
            pipeline = futures[future]
            success, error = future.result()
            results[pipeline] = (success, error)
    
    # Print summary
    tracker.summary()
    
    print(f"\nResults:")
    print(f"  Attack pipeline: {'✓ Success' if results['attack'][0] else '✗ Failed'}")
    print(f"  Honest pipeline: {'✓ Success' if results['honest'][0] else '✗ Failed'}")
    print(f"\nOutput directories:")
    print(f"  Attack: {attack_out_dir}")
    print(f"  Honest: {honest_out_dir}")
    
    # Exit with error if any pipeline failed
    if not all(success for success, _ in results.values()):
        print("\n⚠ Some pipelines failed. Check logs above for details.")
        sys.exit(1)
    
    print("\n✓ All pipelines completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run attack and honest evaluation pipelines in parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments (with sensible defaults)
    parser.add_argument(
        "--conversations", type=str,
        default="datasets/healthbench_consensus_interactions.jsonl",
        help="Input conversations JSONL file"
    )
    
    # Prompt files
    parser.add_argument(
        "--red_prompts", type=str,
        default="prompts/red_interactions.json",
        help="Path to red (adversarial) prompts JSON file"
    )
    parser.add_argument(
        "--blue_prompts", type=str,
        default="prompts/blue_interactions.json",
        help="Path to blue (defender) prompts JSON file"
    )
    parser.add_argument(
        "--honest_prompts", type=str,
        default="prompts/honest.json",
        help="Path to honest prompts JSON file (used instead of red prompts for honest pipeline)"
    )
    
    # Output configuration
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: datasets/experiment_YYYYMMDD_HHMMSS)"
    )
    
    # Sampling configuration
    parser.add_argument(
        "--n", type=int, default=None,
        help="Number of conversations to sample (default: use all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling"
    )
    
    # Model configuration
    parser.add_argument(
        "--red_model", type=str, default="gpt-5",
        help="Model for red team (adversarial) responses"
    )
    parser.add_argument(
        "--blue_model", type=str, default="gpt-4o",
        help="Model for blue team (defender) responses"
    )
    parser.add_argument(
        "--eval_model", type=str, default="gpt-4.1",
        help="Model for evaluations"
    )
    
    args = parser.parse_args()
    main(args)

