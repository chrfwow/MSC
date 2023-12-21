from typing import List

from counterfactuals2 import SearchAlgorithm
from counterfactuals2.Counterfactual import Counterfactual
from counterfactuals2.AbstractTokenizer import AbstractTokenizer
from counterfactuals2.language import Language


class CounterfactualSearch:
    def __init__(self, language: Language, tokenizer: AbstractTokenizer, search_algorithm: SearchAlgorithm):
        self.language = language
        self.tokenizer = tokenizer
        self.search_algorithm = search_algorithm

    def search(self, source_code: str) -> List[Counterfactual]:
        number_of_tokens_in_src, dictionary = self.tokenizer.tokenize(source_code)
        return self.search_algorithm.search(source_code, number_of_tokens_in_src, dictionary)
