import random
from typing import List

from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from counterfactuals2.misc.Counterfactual import Counterfactual
from counterfactuals2.perturber.AbstractPerturber import AbstractPerturber
from counterfactuals2.searchAlgorithms.AbstractSearchAlgorithm import AbstractSearchAlgorithm
from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer
from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker


class KEntry:
    classification: any
    document_indices: List[int]
    masked_indices: List[int]

    def __init__(self, classification: any, document_indices: List[int], masked_indices: List[int] = [], number_of_changes: int = 0):
        self.classification = classification
        self.masked_indices = masked_indices
        self.document_indices = document_indices
        self.number_of_changes = number_of_changes

    def clone(self):
        return KEntry(self.classification, list(self.document_indices), list(self.masked_indices), self.number_of_changes)


def no_duplicate(new_counterfactual: Counterfactual, counterfactuals: List[Counterfactual]):
    for c in counterfactuals:
        if c.code == new_counterfactual.code:
            return False
    return True


class KExpExhaustiveSearch(AbstractSearchAlgorithm):
    def __init__(self, k: int, unmasker: AbstractUnmasker, tokenizer: AbstractTokenizer, classifier: AbstractClassifier,
                 perturber: AbstractPerturber, verbose: bool = False):
        super().__init__(tokenizer, classifier)
        self.k = k
        self.unmasker = unmasker
        self.perturber = perturber
        self.verbose = verbose

    def get_perturber(self) -> AbstractPerturber | None:
        return self.perturber

    def get_unmasker(self) -> AbstractUnmasker | None:
        return self.unmasker

    def perform_search(self, source_code: str, number_of_tokens_in_src: int, dictionary: List[str], original_class: any,
                       original_confidence: float, original_tokens: List[int]) -> List[Counterfactual]:
        random.seed(1)

        original_entry = KEntry(original_class, original_tokens)

        this_iteration: List[KEntry] = [original_entry.clone()]
        next_iteration: List[KEntry] = []
        explanations: List[Counterfactual] = []

        for iteration in range(self.k):
            print("starting iteration", iteration)
            print("searching through up to", len(this_iteration) * len(original_tokens), "mutations")
            for i in range(len(this_iteration)):
                current = this_iteration.pop()

                self.expand(dictionary, current, explanations, original_class, next_iteration)

            print("found", len(explanations), "counterfactuals")

            this_iteration = next_iteration
            next_iteration = []

        return explanations

    def expand(self, dictionary: List[str], entry: KEntry, explanations: List[Counterfactual], original_classification,
               next_iteration: List[KEntry]):

        dict_len = len(dictionary)
        for i in range(len(entry.document_indices)):
            if i not in entry.masked_indices:
                current_doc = entry.clone()
                current_doc.masked_indices.append(i)
                self.perturber.perturb_at_index(i, current_doc.document_indices, dict_len)
                current_doc.number_of_changes += 1

                counterfactual = self.check(dictionary, current_doc, i, original_classification)
                if counterfactual is not None and no_duplicate(counterfactual, explanations):
                    explanations.append(counterfactual)
                    if self.verbose:
                        print("Found a counterfactual", counterfactual)
                else:
                    next_iteration.append(current_doc)

    def check(self, dictionary, entry: KEntry, newly_masked_index: int,
              original_classification, start_time: float, number_of_tokens_in_input: int) -> Counterfactual | None:
        src = self.tokenizer.to_string_unmasked(dictionary, entry.document_indices, newly_masked_index)
        if self.verbose:
            print("checking\n", src)
            print()
        output = self.classifier.classify(src)
        current_classification, score = output[0] if isinstance(output, list) else output
        if current_classification != original_classification:
            return Counterfactual(src, float(score), start_time, number_of_tokens_in_input, entry.number_of_changes, len(entry.document_indices))
        else:
            return None
