from typing import List

from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from counterfactuals2.misc.Counterfactual import Counterfactual
from counterfactuals2.misc.language import Language
from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer


class AbstractSearchAlgorithm:
    def __init__(self, tokenizer: AbstractTokenizer, classifier: AbstractClassifier, language: Language):
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.language = language

    def search(self, source_code: str, number_of_tokens_in_src: int, dictionary: List[str]) -> List[Counterfactual]:
        """Performs the search for counterfactuals. Do not override this function"""
        original_class, original_confidence = self.classifier.classify(source_code)
        original_tokens = []
        for i in range(number_of_tokens_in_src):
            original_tokens.append(i)
        return self.perform_search(source_code, number_of_tokens_in_src, dictionary, original_class,
                                   original_confidence, original_tokens)

    def perform_search(self, source_code: str, number_of_tokens_in_src: int, dictionary: List[str], original_class: any,
                       original_confidence: float, original_tokens: List[int]) -> List[Counterfactual]:
        raise NotImplementedError
