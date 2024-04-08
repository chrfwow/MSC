import random
from typing import List

from counterfactuals2.perturber.AbstractPerturber import AbstractPerturber
from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker


class MaskedPerturber(AbstractPerturber):
    def perturb_in_place(self, source: List[int], dictionary_length: int):
        source[int(len(source) * random.random())] = AbstractUnmasker.MASK_INDEX

    def perturb_at_index(self, index: int, source: List[int], dictionary_length: int):
        source[index] = AbstractUnmasker.MASK_INDEX
