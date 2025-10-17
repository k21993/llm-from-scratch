from abc import ABC
from typing import List, Union 


class Tokenizer(ABC):
    """
    Abstract base class for a tokenizer
    """

    def encode(self, text: Union[List[str], str]):
        raise NotImplementedError
    
    def decode(self, indices: Union[List[List[int]], List[int]]):
        raise NotImplementedError






    