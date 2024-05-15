from typing import List

from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker


class NoOpUnmasker(AbstractUnmasker):
    def get_mask(self) -> None:
        return None

    def get_mask_replacement(self, original_token_id: int, code: str, dictionary: List[str]) -> int:
        return original_token_id
