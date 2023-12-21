import random
from typing import List

from counterfactuals2.AbstractPerturber import AbstractPerturber


class MutationPerturber(AbstractPerturber):
    def perturb_in_place(self, source: List[int], dictionary_length: int):
        what = random.random()
        if what < .3:  # add candidate
            source.insert(int(len(source) * random.random()), int(dictionary_length * random.random()))
        elif what < .6:  # remove candidate
            del source[int(len(source) * random.random())]
        else:  # change candidate
            source[int(len(source) * random.random())] = int(dictionary_length * random.random())
