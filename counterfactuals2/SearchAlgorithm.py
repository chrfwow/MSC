from typing import List

from counterfactuals2.AbstractClassifier import AbstractClassifier
from counterfactuals2.Counterfactual import Counterfactual
from counterfactuals2.AbstractPerturber import AbstractPerturber
from counterfactuals2.AbstractTokenizer import AbstractTokenizer
from counterfactuals2.language import Language


class AbstractSearchAlgorithm:
    def __init__(self, tokenizer: AbstractTokenizer, classifier: AbstractClassifier, perturber: AbstractPerturber,
                 language: Language):
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.perturber = perturber
        self.language = language

    def search(self, source_code: str, number_of_tokens_in_src: int, dictionary: List[str]) -> List[Counterfactual]:
        original_class, original_confidence = self.classifier.classify(source_code)
        original_tokens = []
        for i in range(number_of_tokens_in_src):
            original_tokens.append(i)
        return self.perform_search(source_code, number_of_tokens_in_src, dictionary, original_class,
                                   original_confidence, original_tokens)

    def perform_search(self, source_code: str, number_of_tokens_in_src: int, dictionary: List[str], original_class: any,
                       original_confidence: float, original_tokens: List[int]) -> List[Counterfactual]:
        raise NotImplementedError
