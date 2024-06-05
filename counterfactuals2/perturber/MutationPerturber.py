import random
from typing import List

from counterfactuals2.perturber.AbstractPerturber import AbstractPerturber
from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer


class MutationPerturber(AbstractPerturber):
    def perturb_in_place(self, source: List[int], dictionary_length: int) -> int:
        index = int(len(source) * random.random())
        original = source[index]
        what = random.random()
        if original >= dictionary_length:
            print("no")
        if what < .3:  # add candidate
            source.insert(index, int(dictionary_length * random.random()))
        elif what < .6:  # remove candidate
            del source[index]
        else:  # change candidate
            source[index] = int(dictionary_length * random.random())
        return original

    def perturb_at_index(self, index: int, source: List[int], dictionary_length: int):
        what = random.random()
        if what < 1.0 / len(source):  # remove candidate
            source[index] = AbstractTokenizer.EMPTY_TOKEN_INDEX
        else:  # change candidate
            source[index] = int(dictionary_length * random.random())
