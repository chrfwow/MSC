from typing import List

from counterfactuals2.AbstractTokenizer import AbstractTokenizer


class LineTokenizer(AbstractTokenizer):

    def tokenize(self, source_code: str) -> (int, List[str]):
        result = source_code.split("\n")
        return len(result), result
