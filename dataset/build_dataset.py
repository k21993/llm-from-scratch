import time
import sys
sys.path.append("..")
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union
from concurrent.futures import ProcessPoolExecutor
from tokenizer.bpe import BPETokenizer


def read_data(file: str):
    with open(file, "r") as f:
        text = f.read()
    return text

def chunk_text(text:str, chunk_size:int):
    chunked_text = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunked_text

def tokenize_text(text: str, tokenizer):
    return tokenizer.encode(text)
    
def save_tokenized_text(tokenized_text, save_path):
    #save tokens as a np memmap object
    num_tokens = len(tokenized_text)
    print(f"saving {num_tokens} tokens to {save_path}")
    arr = np.memmap(save_path, dtype=np.uint16, mode="w+", shape=(num_tokens,))
    arr[:] = tokenized_text
    arr.flush()
    print("done")

def build_tokenized_dataset(data_file:str,
                            chunk_size:int=1024,
                            max_workers:int=4,
                            tokenizer=None,
                            save_path="data/tokenized_dataset/tiny_stories/train.dat"
                            ):
    """
    method to save tokenized text data before training. 
    """
    #read and chunk text into blocks of size chunk size for parallel tokenization
    text = read_data(file=data_file)
    chunked_text = chunk_text(text, chunk_size)
    del text

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tokenized_text = list(executor.map(tokenize_text,
                                           chunked_text,
                                           [tokenizer]*len(chunked_text))
                                           )
    tokenized_text = [t for tok in tokenized_text for t in tok]
    save_tokenized_text(tokenized_text=tokenized_text, save_path=save_path)
    return tokenized_text


if __name__ == "__main__":
    
    start = time.time()
    # text = "hey, do you think AI agents are slop? Karpathy thinks so!"
    text_file =  "data/TinyStoriesV2-GPT4-train.txt"
    save_path = "data/tokenized_dataset/tiny_stories/train.dat"
    tok = BPETokenizer(merges_path="tokenizer/.tokenizer_info/merges.txt",
                       vocab_json_path="tokenizer/.tokenizer_info/vocab.json"
                       )
    tokens = build_tokenized_dataset(data_file=text_file,
                                     chunk_size=1024,
                                     max_workers=8,
                                     tokenizer=tok,
                                     save_path=save_path
                                     )
    print("took: ", time.time() - start)
    # print("tokens: ", tokens)
    # print(tok.decode(tokens))
    

    





    
