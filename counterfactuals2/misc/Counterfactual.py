from typing import List

from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer


class Counterfactual:
    def __init__(self, dictionary: List[str], tokens: List[int], score: float):
        self.dictionary = dictionary
        self.tokens = tokens
        self.score = score

    def to_string(self, tokenizer: AbstractTokenizer):
        return "score " + str(self.score) + ": " + tokenizer.to_string(self.dictionary, self.tokens)
