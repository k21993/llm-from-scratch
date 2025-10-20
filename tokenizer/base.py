from abc import ABC
from typing import List, Union 


class Tokenizer(ABC):
    """
    Abstract base class for a tokenizer
    """

    def encode(self, text: str):
        raise NotImplementedError
    
    def decode(self, tokens: List[int]):
        raise NotImplementedError






    