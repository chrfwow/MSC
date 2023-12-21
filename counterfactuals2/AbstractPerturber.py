from typing import List


class AbstractPerturber:
    def perturb(self, source: List[int], dictionary_length: int) -> List[int]:
        result = [*source]
        self.perturb_in_place(result, dictionary_length)
        return result

    def perturb_in_place(self, source: List[int], dictionary_length: int):
        raise NotImplementedError
