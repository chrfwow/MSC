from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker


class NoOpUnmasker(AbstractUnmasker):
    def get_mask(self) -> None:
        return None

    def get_mask_replacement(self, code: str) -> str:
        return code
