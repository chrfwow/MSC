import re
from typing import List

from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer


class LineTokenizer(AbstractTokenizer):

    def tokenize(self, source_code: str) -> (int, List[str]):
        result = re.split("\n", source_code.strip())
        print("tokenization resulted in tokens:")
        i = 0
        for r in result:
            print(i, ":", r)
            i += 1
        return len(result), result
