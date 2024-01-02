from counterfactuals2.perturber.AbstractPerturber import AbstractPerturber
import random
from typing import List


class RemoveTokenPerturber(AbstractPerturber):
    def perturb_in_place(self, source: List[int], dictionary_length: int):
        del source[int(len(source) * random.random())]
