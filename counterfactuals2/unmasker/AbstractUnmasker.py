from typing import List


class AbstractUnmasker:
    MASK_INDEX = -1

    def get_mask(self) -> str | None:
        raise NotImplementedError

    def get_mask_replacement(self, original_token_id: int, code: str, dictionary: List[str]) -> int:
        raise NotImplementedError
