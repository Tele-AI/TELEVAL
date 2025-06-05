import logging
import os
import shutil
import pandas as pd
import tempfile
import atexit
from tqdm import tqdm
from typing import Dict, List
from src.utils import load_and_process_parquet_dataset, TupleJSONLConverter, TupleEncoder
tqdm.pandas()

logger = logging.getLogger(__name__)

class BatchLoader:
    def __init__(self, file, key_col="key", ref_col=None, query_col=None, extra_col=None, 
                 batch_size=1, limit=0, save_query_audio_dir=None):
        self.file = file
        self.key_col = key_col
        self.ref_col = ref_col
        self.query_col = query_col
        self.extra_col = extra_col
        self.batch_size = batch_size
        self.index = 0
        self._temp_dir = None

        if os.path.isfile(file + ".jsonl") or (os.path.isfile(file) and file.endswith(".jsonl")):
            records = self.load_jsonl(file)
        elif isinstance(file, str) and "/" in file:
            print("Using dataset from HuggingFace")
            records = self._process_parquet_records(file, save_query_audio_dir)
        else:
            raise NotImplementedError(f"Unsupported file type: {file}")

        if limit > 0:
            records = records[:limit]
        self.df = pd.DataFrame(records)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.df):
            raise StopIteration
        batch = self.df.iloc[self.index : self.index + self.batch_size].to_dict(orient="records")
        self.index += self.batch_size
        return batch
    
    def __len__(self):
        return len(self.df)

    def _process_parquet_records(self, repo_data_dir, save_audio_root):
        if "/" not in repo_data_dir:
            raise ValueError(f"Invalid input format: {repo_data_dir}, expected 'repo_or_path/data_dir_pattern'")
        parts = repo_data_dir.split("/", 2)
        repo_or_path, data_dir_pattern = "/".join(parts[:2]), parts[-1]  # Tele-AI/TeleSpeech-AudioBench, noise-zh/bubble_-5dB-zh

        if "*.parquet" in data_dir_pattern:
            base_subdir = os.path.normpath(os.path.dirname(data_dir_pattern))
        else:
            base_subdir = os.path.normpath(data_dir_pattern)

        if save_audio_root:
            audio_output_dir = os.path.join(save_audio_root, base_subdir)
            os.makedirs(audio_output_dir, exist_ok=True)
        else:
            self._temp_dir = tempfile.mkdtemp(prefix="temp_audio_")
            atexit.register(self.cleanup)
            audio_output_dir = os.path.join(self._temp_dir, base_subdir)
            os.makedirs(audio_output_dir, exist_ok=True)

        return load_and_process_parquet_dataset(
            repo_or_path, data_dir_pattern, audio_output_dir, key_col=self.key_col
        )

    def cleanup(self):
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    @staticmethod
    def load_jsonl(file):
        return TupleJSONLConverter.load(file)


class BatchSaver:
    def __init__(self, file, overwrite=True):
        self.file = file
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if overwrite and os.path.exists(file):
            print(f"Saving file already exists and will be removed: {file}")
            os.remove(file)

    def save_one(self, item: Dict):
        ext = os.path.splitext(self.file)[1].lower()
        if ext != ".jsonl":
            raise ValueError("Only support jsonl file")
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(TupleJSONLConverter.dumps(item) + "\n")

    def save_all(self, items: List[Dict]):
        ext = os.path.splitext(self.file)[1].lower()
        if ext == ".jsonl":
            with open(self.file, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(TupleJSONLConverter.dumps(item) + "\n")

        elif ext == ".parquet":
            df = pd.DataFrame([TupleEncoder.encode(item) for item in items])
            df.to_parquet(self.file, index=False)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def save_parquet(self, df: pd.DataFrame, columns=None):
        records = df.to_dict(orient="records")
        if columns:
            records = [{k: v for k, v in item.items() if k in columns} for item in records]
        self.save_all(records)
