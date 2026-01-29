
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import glob
import yaml
from typing import Union, List
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils import load_and_process_parquet_dataset, parse_repo_and_pattern

def _download_single_hf_path(path, download_root):
    info = parse_repo_and_pattern(path)
    if info["type"] == "local":
        # local dir, no need of downloading
        return info["repo"]
    repo, subdir = info["repo"], info["pattern"]

    local_dir = os.path.join(download_root, repo.replace("/", "_"))

    repo_local_root = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=[f"{subdir}/*"] if subdir else None,
    )

    if subdir:
        candidate = os.path.join(repo_local_root, subdir)
        return candidate if os.path.isdir(candidate) else repo_local_root
    
    return repo_local_root

def download_from_hf(paths=None, download_root="./hf_downloads", yaml_name=None, max_workers=8):
    """
    Download HuggingFace datasets in parallel.
    paths: None / str / list[str]
    yaml_name: None / str / list[str]
    """
    # auto-load from registry if paths not given
    if paths is None:
        info = collect_yaml_file_info("registry/dataset", yaml_name=yaml_name)
        paths = list(info.values())

    if isinstance(paths, str):
        paths = [paths]

    os.makedirs(download_root, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_download_single_hf_path, p, download_root): p for p in paths}

        for future in as_completed(future_map):
            p = future_map[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"[ERROR] failed to download {p}: {e}")

    return results

def collect_yaml_file_info(yaml_dir, yaml_name: Union[str, List[str]] = None):
    """
    yaml_name: If specified, only this YAML file is read (e.g. "llamaqa.yaml");
               otherwise, all YAML files are read.
    Returns {dataset_name: repo_path}.
    """
    result = {}
    if yaml_name:
        if isinstance(yaml_name, str):
            yaml_name = [yaml_name]
        yaml_files = [os.path.join(yaml_dir, y) for y in yaml_name]
    else:
        yaml_files = glob.glob(os.path.join(yaml_dir, "*.yaml"))

    for yaml_file in yaml_files:
        if not os.path.exists(yaml_file):
            print(f"Error! {yaml_file} not exist!!!", flush=True)
            continue
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for name, config in data.items():
            file_path = config.get("args", {}).get("file", None)
            result[name] = file_path

    return result

def export_parquet_to_jsonl(paths=None, save_root_dir="./", yaml_name=None, auto_download=False):
    if auto_download:
        paths = download_from_hf(paths, yaml_name=yaml_name)
    else:
        if not paths:
            paths = list(collect_yaml_file_info("registry/dataset", yaml_name=yaml_name).values())
    
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        info = parse_repo_and_pattern(path)
        repo_or_path, pattern = info["repo"], info["pattern"]
        is_local = info["type"] == "local"
        if is_local:
            base_subdir = os.path.basename(os.path.normpath(repo_or_path))
        else:
            base_subdir = os.path.basename(os.path.normpath(pattern))
        
        jsonl_path = os.path.join(save_root_dir, base_subdir, base_subdir + ".jsonl")
        audio_output_dir = os.path.join(save_root_dir, base_subdir, "audios")
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

        records = load_and_process_parquet_dataset(
            repo_or_path, 
            pattern,
            audio_output_dir,
            key_col="key",
            is_local=is_local,
            tuple_decode=False
        )

        with open(jsonl_path, "w", encoding="utf-8") as fout:
            for r in records:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"JSONL saved to: {jsonl_path}")
        print(f"Audio files saved under: {audio_output_dir}")

if __name__ == "__main__":
    save_root_dir = "./"
    
    print("1. Download and process all datasets")
    print("2. Download and process specific YAML files (aqa.yaml, human.yaml)")
    print("3. Download from specified sources only")
    print("4. Process local parquet files only")
    choice = "1"
    
    if choice == "1":
        # 1. Download and process all datasets
        export_parquet_to_jsonl(save_root_dir=save_root_dir, auto_download=True)
    elif choice == "2":
        # 2. Download and process specific YAML files
        export_parquet_to_jsonl(yaml_name=["aqa.yaml", "human.yaml"], save_root_dir=save_root_dir, auto_download=True)
    elif choice == "3":
        # 3. Download from specified sources only
        export_parquet_to_jsonl(paths=[
            "Tele-AI/TELEVAL/age-zh",
            "Tele-AI/TELEVAL/llamaqa-zh"
        ], save_root_dir=save_root_dir, auto_download=True)
    elif choice == "4":
        # 4. Parquet files already exist locally; process only 
        export_parquet_to_jsonl(paths="/path/to/llamaqa-zh", save_root_dir=save_root_dir)
    