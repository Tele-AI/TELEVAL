import os
import json
import argparse
import shutil
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="res")
    parser.add_argument("--transpose", default=True)
    return parser.parse_args()

def main():
    results = dict()
    args = get_args()
    summary_dir = os.path.join(args.root_dir, "summary")
    column_order = []
    for model_name in os.listdir(summary_dir):
        model_path = os.path.join(summary_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        results[model_name] = dict()

        jsonl_files = [f for f in os.listdir(model_path) if f.endswith(".jsonl")]
        jsonl_files.sort()


        for jsonl_file in jsonl_files:
            dataset_name = os.path.splitext(jsonl_file)[0]
            if dataset_name not in column_order:
                column_order.append(dataset_name)

            file_path = os.path.join(model_path, jsonl_file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    score_str = ""
                    line = f.readline().strip()
                    data = json.loads(line)
                    for key, value in data.items():
                        score_str += value + " "
                    results[model_name][dataset_name] = score_str
            except Exception as e:
                print(f"fail to read {file_path}: {e}")
                raise e

    df = pd.DataFrame.from_dict(results, orient="index")
    df = df.reindex(columns=column_order)

    if args.transpose:
        df = df.T
    df.to_csv(f"{args.root_dir}/results.csv", encoding="utf-8")

    print("========================== results ==========================", flush=True)
    terminal_width = shutil.get_terminal_size().columns
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", terminal_width)
    print(df)

if __name__ == "__main__":
    main()