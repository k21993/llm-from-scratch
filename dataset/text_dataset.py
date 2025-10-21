import os
from typing import List, Dict, Tuple
import logging
import math
import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer.bpe import BPETokenizer

class TextDataTokenizer(Dataset):
    """
    Pytorch dataset class to load a saved tokenized data file (.npy).
    """
    def __init__(self, np_file_path:str, split:str="train", seq_len:int=512, pad_token_id:int=0):
        self.split = split
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

        if split == "train":
            self.data = np.memmap(os.path.join(np_file_path, "train.dat"), dtype=np.uint16, mode="r")
        elif split == "val":
            self.data = np.memmap(os.path.join(np_file_path, "val.dat"), dtype=np.uint16, mode="r")
        else:
            raise ValueError("split has to be one of [train, val], got: ", split)
        self.data_len = len(self.data)
        logging.info(f"loaded {split } with {self.data_len} tokens")
        
    def __len__(self):
        #ceil since we don't want to skip the last index in the data
        return math.ceil(self.data_len / self.seq_len)
    
    def __getitem__(self, idx):
        start = idx*self.seq_len
        end = min(start + self.seq_len, self.data_len)
        seq = self.data[start:end]
        
        #pad in case the length of seq is < seq_len
        if len(seq) < self.seq_len:
            pad = np.zeros(self.seq_len - len(seq), self.pad_token_id, dtype=np.uint16)
            seq = np.concatenate([seq, pad])
        
        #TODO: benchmark moving to GPU and upcasting to long vs upcasting first.
        return torch.from_numpy(seq.astype(np.int64)) 
    
        