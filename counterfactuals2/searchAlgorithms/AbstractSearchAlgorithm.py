import time
from typing import List

from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from counterfactuals2.misc.Counterfactual import Counterfactual
from counterfactuals2.misc.SearchResults import SearchResult
from counterfactuals2.perturber.AbstractPerturber import AbstractPerturber
from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer
from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker


class AbstractSearchAlgorithm:
    def __init__(self, tokenizer: AbstractTokenizer, classifier: AbstractClassifier):
        self.tokenizer = tokenizer
        self.classifier = classifier

    def search(self, source_code: str) -> SearchResult:
        """Performs the search for counterfactuals. Do not override this function"""
        start = time.time()
        number_of_tokens_in_src, dictionary = self.tokenizer.tokenize(source_code)
        original_class, original_confidence = self.classifier.classify(source_code)

        print("input classified as", original_class, "with a confidence of", original_confidence)

        original_tokens = []
        for i in range(number_of_tokens_in_src):
            original_tokens.append(i)
        result = self.perform_search(source_code, number_of_tokens_in_src, dictionary, original_class,
                                     original_confidence, original_tokens)
        end = time.time()
        print("search took", end - start, "seconds")
        return SearchResult(source_code, result, self, self.classifier, self.get_perturber(), self.tokenizer, self.get_unmasker(), end - start)

    def get_perturber(self) -> AbstractPerturber | None:
        return None

    def get_unmasker(self) -> AbstractUnmasker | None:
        return None

    def perform_search(self, source_code: str, number_of_tokens_in_src: int, dictionary: List[str], original_class: any,
                       original_confidence: float, original_tokens: List[int]) -> List[Counterfactual]:
        raise NotImplementedError
