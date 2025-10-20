import os
from typing import List, Dict, Tuple
from torch.utils.data import Dataset

from tokenizer.bpe import BPETokenizer

class TextDataTokenizer(Dataset):
    """
    Pytorch dataset class to load a saved tokenized data file (.npy).
    """
    def __init__(self, files:List[str], tokenizer:BPETokenizer, ):
        pass
        
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
