from typing import List


class AbstractPerturber:
    def perturb_in_place(self, source: List[int], dictionary_length: int) -> int:
        """Returns the original value of the changed entry before change"""
        raise NotImplementedError

    def perturb_at_index(self, index: int, source: List[int], dictionary_length: int):
        raise NotImplementedError
