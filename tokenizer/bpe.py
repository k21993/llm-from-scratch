

from typing import List, Union
from base import Tokenizer

class BPETokenizer(Tokenizer):
    """
    Class which implements BPE tokenizer.
    """

    def __init__(self):
        pass

    def encode(self, text: Union[List[str], str]):
        pass

    def decode(self, indices: Union[List[List[int]], List[int]]):
        pass



if __name__ == "__main__":
    text = "hey!"
    tok = BPETokenizer()
    tok.encode(text=text)