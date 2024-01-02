class AbstractUnmasker:
    MASK_INDEX = -1

    def get_mask(self) -> str:
        raise NotImplementedError

    def get_mask_replacement(self, code: str) -> str:
        raise NotImplementedError
