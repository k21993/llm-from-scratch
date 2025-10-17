import os
import json
from tqdm import tqdm
from typing import List, Dict, Tuple, Union
import heapq
import regex as re
import time
from glob import glob


#globals
SAVE_PATH = ".tokenizer_info"
os.makedirs(".tokenizer_info", exist_ok=True)

#data helper
def load_tiny_stories(path:str) -> str:
    # getting OOM
    files  = glob(os.path.join(path,"Tiny*"))
    text = ""
    for file in tqdm(files):
        with open(file, "r") as f:
            text += f.read()
    
    return text

class BPETrainer:
    def __init__(self, num_merges: int, text: str):
        self.num_merges = num_merges
        self.text = text
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.pre_tokens = {}
        self.merges = []

    def get_pre_tokens(self) -> None:
        #split by space (simple v0)
        # words = self.text.split(" ")

        # GPT-2 / TikToken-style regex pattern
        pattern = re.compile(
            r"(?:'?[sdmt]|'ll|'ve|'re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            re.UNICODE
    )
        # Find all regex matches in the text
        words = pattern.findall(self.text)

        # For debugging: show what’s getting split (optional)
        print("Sample pre-tokens:", words[:30])

        for w in words:
            b = list(w.encode("utf-8"))
            if w not in self.pre_tokens:
                self.pre_tokens[w] = {"count": 1, "token_idxs": b}
            else:
                self.pre_tokens[w]["count"] += 1

    def count_pairs(self, updated_tokens) -> Dict[Tuple[int, int], int]:
        pairs = {}
        for t, info in self.pre_tokens.items():
            if updated_tokens is not None and t not in updated_tokens:
                continue
            count = info["count"]
            tok = info["token_idxs"]
            for i in range(len(tok) - 1):
                pair = (tok[i], tok[i + 1])
                pairs[pair] = pairs.get(pair, 0) + count
        return pairs

    def merge_pair(self, pair: Tuple[int, int], new_id: int) -> set:
        updated_tokens = set()
        for t, info in self.pre_tokens.items():
            tok = info["token_idxs"]
            new_tok = []
            i = 0
            while i < len(tok):
                if i < len(tok) - 1 and (tok[i], tok[i + 1]) == pair:
                    new_tok.append(new_id)
                    updated_tokens.add(t)
                    i += 2
                else:
                    new_tok.append(tok[i])
                    i += 1
            info["token_idxs"] = new_tok
        
        return updated_tokens
    
    @staticmethod
    def _get_max_pair(pairs):
        if pairs:
            return max(pairs, key=pairs.get)

    def train(self):
        tic = time.time()
        self.get_pre_tokens()
        updated_tokens = None
        for _ in tqdm(range(self.num_merges)):
            pairs = self.count_pairs(updated_tokens)
            #add this to avoid running out of pairs very early in the training. 
            if not pairs and updated_tokens is not None:
                pairs = self.count_pairs(None)

            max_pair = BPETrainer._get_max_pair(pairs)
            if not max_pair:
                break
            print("max_pair: ", max_pair)
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
            self.merges.append(max_pair)
            updated_tokens = self.merge_pair(max_pair, new_id)
        
        # import pdb; pdb.set_trace()

        self.save()
        print("done. took: ", time.time() - tic)

    def save(self):
        vocab_str = {v.decode("latin-1"): i for i, v in self.vocab.items()}
        merges_str = [(self.vocab[a].decode("latin-1"), self.vocab[b].decode("latin-1"))
                      for a, b in self.merges]

        with open(os.path.join(SAVE_PATH, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_str, f, ensure_ascii=False, indent=2)

        with open(os.path.join(SAVE_PATH, "merges.txt"), "w", encoding="utf-8") as f:
            for a, b in merges_str:
                f.write(f"{a} {b}\n")

        print(f"✅ Saved tokenizer files to {SAVE_PATH}")

if __name__ == "__main__":
    # text = "banana band banana"
    # text = "the she there hero"
    text = load_tiny_stories("/Users/karthik/Desktop/Karthik/projects/cs336/assignment1-basics/data/")
    bpe = BPETrainer(num_merges=7000, text=text)
    bpe.train()
