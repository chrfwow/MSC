import time
from typing import List

from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from counterfactuals2.misc.Counterfactual import Counterfactual
from counterfactuals2.misc.SearchParameters import SearchParameters
from counterfactuals2.misc.SearchResults import SearchResult, SearchError, InvalidClassificationResult
from counterfactuals2.perturber.AbstractPerturber import AbstractPerturber
from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer
from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker


class AbstractSearchAlgorithm:
    def __init__(self, tokenizer: AbstractTokenizer, classifier: AbstractClassifier, verbose: bool):
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.verbose = verbose

    def search(self, source_code: str) -> SearchResult:
        """Performs the search for counterfactuals. Do not override this function"""
        start = time.time()
        try:

            number_of_tokens_in_src, dictionary = self.tokenizer.tokenize(source_code)
            original_class, original_confidence = self.classifier.classify(source_code)

            if self.verbose:
                print("input classified as", original_class, "with a confidence of", original_confidence)

            if original_class:
                return InvalidClassificationResult(source_code, original_class, self, self.classifier, self.get_perturber(), self.tokenizer, self.get_unmasker(), 0, self.get_parameters())

            original_tokens = []
            for i in range(number_of_tokens_in_src):
                original_tokens.append(i)
            result = self.perform_search(source_code, number_of_tokens_in_src, dictionary, original_class,
                                         original_confidence, original_tokens)
            end = time.time()
            if self.verbose:
                print("search took", end - start, "seconds")
            return SearchResult(source_code, result, self, self.classifier, self.get_perturber(), self.tokenizer, self.get_unmasker(), end - start, self.get_parameters())
        except Exception as e:
            if self.verbose:
                print(e)
            end = time.time()
            return SearchError(source_code, e, self, self.classifier, self.get_perturber(), self.tokenizer, self.get_unmasker(), end - start, self.get_parameters())

    def get_parameters(self) -> SearchParameters:
        raise NotImplementedError

    def get_perturber(self) -> AbstractPerturber | None:
        return None

    def get_unmasker(self) -> AbstractUnmasker | None:
        return None

    def perform_search(self, source_code: str, number_of_tokens_in_src: int, dictionary: List[str], original_class: any,
                       original_confidence: float, original_tokens: List[int]) -> List[Counterfactual]:
        raise NotImplementedError
