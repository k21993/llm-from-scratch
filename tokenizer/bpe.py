import json
from typing import List, Union, Dict
from tokenizer.base import Tokenizer
from collections import defaultdict

class BPETokenizer(Tokenizer):
    """
    Class which implements BPE tokenizer.
    """ 

    def __init__(self, merges_path:str, vocab_json_path:str, sep="#&#"):
        with open(merges_path, "r") as f:
            self.merges = f.readlines()
                
        with open(vocab_json_path, "r") as f:
            self.vocab = json.load(f)

        self.token_to_str = {i:v for v,i in self.vocab.items()}
        #convert merges to ordered pairs: [[byte_pair1, byte_pair2], [..], ...]
        self.merges = self._prepare_merges(sep)


    def _prepare_merges(self, sep="#&#"):
        cleaned = []
        for m in self.merges:
            parts = m.rstrip("\n").split(sep)
            if len(parts) != 2:
                continue
            a, b = parts
            # they're already strings
            if a in self.vocab and b in self.vocab:
                cleaned.append((self.vocab[a], self.vocab[b]))
            else:
                print(f"⚠️ Skipping unknown merge ({a}, {b})")
        return cleaned

    @staticmethod
    def _get_pairs(tokens:List[int]) -> Dict[tuple, List]:
        pairs = defaultdict(list)
        for i in range(len(tokens)-1):
            pair = (tokens[i], tokens[i+1])
            pairs[pair].append(i)
        
        return pairs
    
    def _merge_pair(self, raw_tokens:List[int], pair:tuple , pairs:Dict[tuple, List]):
        pair_idxs = pairs[pair]
        merged_txt = self.token_to_str[pair[0]] + self.token_to_str[pair[1]]
        if not merged_txt in self.vocab: #some tokens like " " + " " are not found!
            print("str not found in vocab: ", merged_txt, "ids: ", pair)
            return raw_tokens

        new_token = self.vocab[merged_txt]
        for idx in pair_idxs:
            raw_tokens[idx] = new_token
            raw_tokens[idx + 1] = -1 #dummy token to be deleted
        
        return raw_tokens

    def encode(self, text: str):
        #(O(num_merges)x(num_tokens))
        tokens = list(text.encode("utf-8"))
        pairs = BPETokenizer._get_pairs(tokens=tokens)
        for merge in self.merges:
            if merge in pairs:
                tokens = self._merge_pair(tokens,merge,pairs)
                pairs = BPETokenizer._get_pairs(tokens)
        
        #remove idx+1 tokens which were merged now
        tokens = [r for r in tokens if r!= -1]
        
        return tokens
        
    def decode(self, tokens: List[int]) -> str:
        out = [self.token_to_str[t] for t in tokens]
        return "".join(out)

if __name__ == "__main__":
    text = "yo whaddup"
    tok = BPETokenizer(merges_path=".tokenizer_info/merges.txt",
                       vocab_json_path=".tokenizer_info/vocab.json"
                       )
    tokens = tok.encode(text=text)
    decoded_text = tok.decode(tokens=tokens)

    print(tokens)
    print(decoded_text)