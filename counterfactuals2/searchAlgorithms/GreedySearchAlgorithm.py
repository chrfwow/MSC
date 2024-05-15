import random
import time
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
    confidence_delta: float = 0

    def __init__(self, classification: any, document_indices: List[int], number_of_changes: int = 0):
        self.classification = classification
        self.document_indices = document_indices
        self.number_of_changes = number_of_changes

    def clone(self):
        return KEntry(self.classification, list(self.document_indices), self.number_of_changes)

    def __lt__(self, other):  # <, exactly inverse because python knows only a min heap, and we want a max heap
        return self.confidence_delta >= other.confidence_delta

    def __le__(self, other):  # <=
        return self.confidence_delta > other.confidence_delta


def no_duplicate(new_counterfactual: Counterfactual, counterfactuals: List[Counterfactual]):
    for c in counterfactuals:
        if c.code == new_counterfactual.code:
            return False
    return True


class GreedySearchAlgorithm(AbstractSearchAlgorithm):
    def __init__(self, max_search_steps: int, unmasker: AbstractUnmasker, tokenizer: AbstractTokenizer,
                 classifier: AbstractClassifier, perturber: AbstractPerturber, verbose: bool = False, max_age: int = 25,
                 max_survivors: int = 10):
        super().__init__(tokenizer, classifier)
        self.max_search_steps = max_search_steps
        self.unmasker = unmasker
        self.perturber = perturber
        self.verbose = verbose
        self.max_age = max_age
        self.max_survivors = max_survivors

    def get_perturber(self) -> AbstractPerturber | None:
        return self.perturber

    def get_unmasker(self) -> AbstractUnmasker | None:
        return self.unmasker

    def perform_search(self, source_code: str, number_of_tokens_in_src: int, dictionary: List[str], original_class: any,
                       original_confidence: float, original_tokens: List[int]) -> List[Counterfactual]:
        random.seed(1)
        original_dictionary_length = len(dictionary)

        original_entry = KEntry(original_class, original_tokens)

        counterfactuals: List[Counterfactual] = []
        pool: List[KEntry] = []

        start_time = time.time()

        for i in range(number_of_tokens_in_src):
            current = original_entry.clone()
            self.perturber.perturb_at_index(i, current.document_indices, len(dictionary))
            current.number_of_changes += 1

            result = self.check(dictionary, current, original_class)

            if type(result) == Counterfactual:
                counterfactuals.append(result)
            else:
                delta = original_confidence - result
                current.confidence_delta = delta
                pool.append(current)
                if self.verbose:
                    print("#", i, "added with a delta of", delta)

        for step in range(self.max_search_steps - number_of_tokens_in_src):
            if len(pool) == 0:
                print("search completed, no candidates left")
                return counterfactuals
            best_index = self.get_roulette_best_index(pool)
            best = pool[best_index]

            if self.verbose:
                print("#", step + number_of_tokens_in_src, "popped the best with a delta of", best.confidence_delta)

            current = best.clone()
            self.perturber.perturb_in_place(current.document_indices, len(dictionary))
            current.number_of_changes += 1

            result = self.check(dictionary, current, original_class, start_time, original_dictionary_length)
            if type(result) == Counterfactual:
                del pool[best_index]
                if no_duplicate(result, counterfactuals):
                    counterfactuals.append(result)
            else:
                delta = original_confidence - result
                current.confidence_delta = delta
                pool.append(current)

        print("search completed, search limit reached")
        return counterfactuals

    def get_roulette_best_index(self, pool: List[KEntry]) -> int:
        min = 999999999
        sum = 0
        for e in pool:
            sum += e.confidence_delta
            if e.confidence_delta < min:
                min = e.confidence_delta

        if min < 0:
            min = abs(min)
            sum += len(pool) * min
        else:
            min = 0

        trigger = random.random() * sum
        sum = 0
        for i in range(len(pool)):
            current = pool[i]
            sum += current.confidence_delta + min
            if sum > trigger:
                return i
        return len(pool) - 1

    def check(self, dictionary, entry: KEntry, original_classification, start_time: float, number_of_tokens_in_input: int) -> Counterfactual | float:
        src = self.tokenizer.to_string_unmasked(dictionary, entry.document_indices)
        if self.verbose:
            print("checking\n", src)
            print()
        output = self.classifier.classify(src)
        current_classification, score = output[0] if isinstance(output, list) else output
        if current_classification != original_classification:
            if self.verbose:
                print("^^^^^^^ was a counterfactual")
            return Counterfactual(src, float(score), start_time, number_of_tokens_in_input, entry.number_of_changes, len(entry.document_indices))
        else:
            return float(score)
