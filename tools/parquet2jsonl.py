
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import glob
import yaml
from src.utils import load_and_process_parquet_dataset

def collect_yaml_file_info(yaml_dir):
    result = {}
    yaml_files = glob.glob(os.path.join(yaml_dir, "*.yaml"))

    for yaml_file in yaml_files:
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for name, config in data.items():
            file_path = config.get("args", {}).get("file", None)
            result[name] = file_path

    return result

def export_parquet_to_jsonl(repo_or_path="Tele-AI/TeleSpeech-AudioBench", data_dir_pattern="llamaqa-zh", save_root_dir="./", is_local=False):
    print(f"processing {repo_or_path}, {data_dir_pattern} data from huggingface, saving to {save_root_dir}")
    if "*.parquet" in data_dir_pattern:
        base_subdir = os.path.normpath(os.path.dirname(data_dir_pattern))
    else:
        base_subdir = os.path.normpath(data_dir_pattern)

    jsonl_filename = os.path.basename(base_subdir) + ".jsonl"
    jsonl_path = os.path.join(save_root_dir, base_subdir, jsonl_filename)
    audio_output_dir = os.path.join(save_root_dir, "audios", base_subdir)

    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    if os.path.exists(jsonl_path):
        print(f"JSONL already exists and will be overwritten: {jsonl_path}")

    records = load_and_process_parquet_dataset(
        repo_or_path, data_dir_pattern, audio_output_dir, key_col="key", is_local=is_local, tuple_decode=False
    )

    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for record in records:
            print(json.dumps(record, ensure_ascii=False), file=fout)

    print(f"JSONL saved to: {jsonl_path}")
    print(f"Audio files saved under: {audio_output_dir}")

if __name__ == "__main__":
    save_root_dir = "./"
    all_dataset = collect_yaml_file_info("../registry/dataset")
    for dataset, repo_data_dir in all_dataset.items():
        parts = repo_data_dir.split("/", 2)
        repo_or_path, data_dir_pattern = "/".join(parts[:2]), parts[-1]
        print("repo_or_path, data_dir_pattern: ", repo_or_path, data_dir_pattern)
        # export_parquet_to_jsonl(repo_or_path, data_dir_pattern, save_root_dir)
        export_parquet_to_jsonl(data_dir_pattern="chinese_quiz-zh")  # noise-zh/bubble_-5dB-zh
        raise RuntimeError