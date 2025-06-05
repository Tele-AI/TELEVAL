import os
import logging
import sys
import debugpy
import torchaudio
import torch
import numpy as np

def set_logging():
    rank = os.environ.get("RANK", 0)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format=f"%(asctime)s [RANK {rank}] (%(module)s:%(lineno)d) %(levelname)s : %(message)s",
    )
    
def waiting_for_debug(ip, port):
    rank = os.environ.get("RANK", "0")
    debugpy.listen((ip, port))
    logging.info(f"[rank = {rank}] Waiting for debugger attach...")
    debugpy.wait_for_client()
    logging.info(f"[rank = {rank}] Debugger attached")
    
def load_audio(audio_path, target_sample_rate):
    wav, raw_sample_rate = torchaudio.load(audio_path) # (1, T)   tensor 
    if raw_sample_rate != target_sample_rate:   
        wav = torchaudio.functional.resample(wav, raw_sample_rate, target_sample_rate) # tensor 
    wav = np.expand_dims(wav.squeeze(0).numpy(), axis=1)
    wav = torch.tensor(wav).reshape(1, 1, -1)
    return wav

def save_audio(audio_outpath, audio_out, sample_rate):
    print(audio_outpath, audio_out, sample_rate)
    torchaudio.save(
        audio_outpath, 
        audio_out, 
        sample_rate=sample_rate, 
        encoding='PCM_S', 
        bits_per_sample=16
    )
    logging.info(f"success save audio at {audio_outpath}")