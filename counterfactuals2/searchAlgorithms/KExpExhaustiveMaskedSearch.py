import random
from typing import List, Tuple

from common.code_formatter import format_code
from counterfactuals2 import counterfactual_search
from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from counterfactuals2.misc.Counterfactual import Counterfactual
from counterfactuals2.misc.language import Language
from counterfactuals2.perturber.AbstractPerturber import AbstractPerturber
from counterfactuals2.searchAlgorithms.AbstractSearchAlgorithm import AbstractSearchAlgorithm
from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer
from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker


class KEntry:
    classification: any
    document_indices: List[int]
    masked_indices: List[int]

    def __init__(self, classification: any, document_indices: List[int], masked_indices: List[int] = []):
        self.classification = classification
        self.masked_indices = masked_indices
        self.document_indices = document_indices

    def clone(self):
        return KEntry(self.classification, list(self.document_indices), list(self.masked_indices))


class KExpExhaustiveMaskedSearch(AbstractSearchAlgorithm):
    def __init__(self, k: int, unmasker: AbstractUnmasker, tokenizer: AbstractTokenizer, classifier: AbstractClassifier,
                 language: Language):
        super().__init__(tokenizer, classifier, language)
        self.k = k
        self.unmasker = unmasker

    def perform_search(self, source_code: str, number_of_tokens_in_src: int, dictionary: List[str], original_class: any,
                       original_confidence: float, original_tokens: List[int]) -> List[Counterfactual]:
        random.seed(1)
        dictionary_length = len(dictionary)

        original_entry = KEntry(original_class, original_tokens)

        this_iteration: List[KEntry] = [original_entry.clone()]
        next_iteration: List[KEntry] = []
        explanations: List[Counterfactual] = []

        for iteration in range(self.k):
            print("starting iteration", iteration)
            for i in range(len(this_iteration)):
                current = this_iteration.pop()

                self.expand(dictionary, current, explanations, original_class, next_iteration)

            this_iteration = next_iteration
            next_iteration = []

        return explanations

    def expand(self, dictionary: List[str], entry: KEntry, explanations: List[Counterfactual], original_classification,
               next_iteration: List[KEntry]):

        for i in range(len(entry.document_indices)):
            if i not in entry.masked_indices:
                current_doc = entry.clone()
                current_doc.masked_indices.append(i)
                current_doc.document_indices[i] = AbstractUnmasker.MASK_INDEX

                counterfactual = self.check(dictionary, current_doc, i, original_classification)
                if counterfactual is not None:
                    explanations.append(counterfactual)
                else:
                    next_iteration.append(current_doc)

    def check(self, dictionary, entry: KEntry, newly_masked_index: int,
              original_classification) -> Counterfactual | None:
        src = self.tokenizer.to_string_unmasked(dictionary, entry.document_indices, newly_masked_index)
        initial_output = self.classifier.classify(src)
        current_classification, unused = initial_output[0] if isinstance(initial_output, list) else \
            initial_output
        if current_classification != original_classification:
            return Counterfactual(src, 1)
        else:
            return None
