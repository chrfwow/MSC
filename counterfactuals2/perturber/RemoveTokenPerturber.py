import random
from typing import List

from counterfactuals2.perturber.AbstractPerturber import AbstractPerturber
from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer


class RemoveTokenPerturber(AbstractPerturber):
    def perturb_in_place(self, source: List[int], dictionary_length: int) -> int:
        index = int(len(source) * random.random())
        original = source[index]
        del source[index]
        return original

    def perturb_at_index(self, index: int, source: List[int], dictionary_length: int):
        source[index] = AbstractTokenizer.EMPTY_TOKEN_INDEX
