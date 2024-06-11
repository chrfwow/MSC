import re
from typing import List

from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer


class LineTokenizer(AbstractTokenizer):

    def tokenize(self, source_code: str, verbose: bool = False) -> (int, List[int], List[str]):
        result = re.split("\n+", source_code)

        d = dict()  # [str] = index
        indices = []
        current_index = 0

        for r in result:
            r = r.strip()
            if r in d.keys():
                indices.append(d[r])
            else:
                d[r] = current_index
                indices.append(current_index)
                current_index += 1

        if verbose:
            print("tokenization resulted in tokens:")
            i = 0
            for r in result:
                print(i, ":", r)
                i += 1
        return len(indices), indices, list(d.keys())

    def get_joining_string(self) -> str:
        """Returns the string used to join the list of tokens when converting tokens to strings"""
        return "\n"
