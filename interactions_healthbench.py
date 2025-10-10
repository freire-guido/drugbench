import argparse
import json
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from simple_evals import common
from simple_evals.healthbeanch_eval_interactions import HealthBenchEval
from simple_evals.sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)
from simple_evals.sampler.interactions_sampler import InteractionsSampler

load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--red_model",
        type=str,
        help="Select a model by name",
    )
    parser.add_argument(
        "--blue_model",
        type=str,
        help="Select a model by name",
    )
    parser.add_argument(
        "--interactions",
        type=str,
        default="datasets/interactions.csv",
        help="Path to JSON file containing interactions for each example.",
    )
    parser.add_argument(
        '--red_messages',
        type=str,
        default="prompts/red_interactions.json",
        help="Path to JSON file containing red team messages.",
    )
    parser.add_argument(
        '--blue_messages',
        type=str,
        default="prompts/blue_messages.json",
        help="Path to JSON file containing blue team messages.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=120,
        help="Number of threads to run. Only supported for HealthBench and HealthBenchMeta.",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )

    args = parser.parse_args()

    # Load red and blue messages from JSON files
    with open(args.red_messages, 'r') as f:
        red_messages = json.load(f)
    
    with open(args.blue_messages, 'r') as f:
        blue_messages = json.load(f)

    models = {
        "gpt-5": "gpt-5-2025-08-07",
        "gpt-5-mini": "gpt-5-mini-2025-08-07",
        "gpt-5-nano": "gpt-5-nano-2025-08-07",
        "gpt-4.1": "gpt-4.1-2025-04-14",
        "gpt-4o": "gpt-4o-2024-08-06",
    }

    sampler = InteractionsSampler(
        interactions_path=args.interactions,
        red_model=models[args.red_model],
        blue_model=models[args.blue_model],
        reasoning_model=True,
        red_messages=red_messages,
        blue_messages=blue_messages,
    )

    print(f"Running with args {args}")

    grading_sampler = ChatCompletionSampler(
        model="gpt-4.1-2025-04-14",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    )

    eval_name = "healthbench"
    eval_obj = HealthBenchEval(
        grader_model=grading_sampler,
        num_examples=10 if args.debug else args.examples,
        # n_repeats=args.n_repeats or 1,
        n_threads=args.n_threads or 1,
        subset_name="consensus",
    )

    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    result = eval_obj(sampler)
    # ^^^ how to use a sampler
    file_stem = f"{eval_name}_{models[args.red_model]}_{models[args.blue_model]}"
    # file stem should also include the year, month, day, and time in hours and minutes
    file_stem += f"_{date_str}"
    report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
    print(f"Writing report to {report_filename}")
    with open(report_filename, "w") as fh:
        fh.write(common.make_report(result))
    assert result.metrics is not None
    metrics = result.metrics | {"score": result.score}
    # Sort metrics by key
    metrics = dict(sorted(metrics.items()))
    print(metrics)
    result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
    with open(result_filename, "w") as f:
        f.write(json.dumps(metrics, indent=2))
    print(f"Writing results to {result_filename}")

    full_result_filename = f"/tmp/{file_stem}{debug_suffix}_allresults.json"
    with open(full_result_filename, "w") as f:
        result_dict = {
            "score": result.score,
            "metrics": result.metrics,
            "htmls": result.htmls,
            "convos": result.convos,
            "metadata": result.metadata,
        }
        f.write(json.dumps(result_dict, indent=2, default=str))
        print(f"Writing all results to {full_result_filename}")

    mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
