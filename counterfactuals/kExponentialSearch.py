import random
from typing import Tuple, List, Set

from counterfactuals.ReplaceWordsPerturbation_plbart import ReplaceWordsPerturbationPlBart
from misc.code_formatter import format_code
from counterfactuals.counterfactual_search import BaseCounterfactualSearch


class KEntry:
    indices: Set[int] = {}
    document: List[int] = []

    def clone(self):
        a = KEntry()
        a.indices = {*self.indices}
        a.document = [*self.document]
        return a


class KExponentialSearch(BaseCounterfactualSearch):
    def __init__(self, language, k: int = 3):
        self.language = language
        self.proxy = ReplaceWordsPerturbationPlBart(language)
        self.k = k

    def search(self, document):
        print("searching for counterfactuals for\n", format_code(document, self.language))
        proxy = self.proxy

        random.seed(1)

        document_length, dictionary = proxy.document_to_perturbation_space(document)
        print("document_length", document_length, "dictionary", dictionary)

        dictionary_length = len(dictionary)
        explanations = []
        perturbation_tracking = []

        original_document_indices: List[int] = []
        for i in range(document_length):
            original_document_indices.append(i)

        initial_output = proxy.classify(
            format_code(proxy.perturb_positions(dictionary, original_document_indices), self.language))
        print("initial_output", initial_output)

        initial_classification, initial_score = initial_output[0] if isinstance(initial_output, list) else \
            initial_output
        print("initial_classification", initial_classification, initial_score)

        original_entry = KEntry()
        original_entry.document = [*original_document_indices]

        self.search_iterative(original_entry, document_length, dictionary, dictionary_length,
                              initial_classification, explanations, perturbation_tracking, self.k, self.language)
        return document, explanations, perturbation_tracking, dictionary, document_length

    def search_iterative(self, entry: KEntry, document_length: int, dictionary: List[str], dictionary_length: int,
                         original_classification, explanations: List, perturbation_tracking: List, k: int,
                         language: str):
        this_iteration: List[KEntry] = []
        next_iteration: List[KEntry] = []

        # self.expand(dictionary, dictionary_length, entry, document_length, explanations, original_classification,perturbation_tracking, this_iteration, language)

        this_iteration.append(entry.clone())

        for iteration in range(k):
            print("starting iteration", iteration)
            for i in range(len(this_iteration)):
                current = this_iteration.pop()

                for j in range(document_length):
                    if j not in current.indices:
                        self.expand(dictionary, dictionary_length, current, explanations, original_classification,
                                    perturbation_tracking, next_iteration, language, j)

            this_iteration = next_iteration
            next_iteration = []

    def expand(self, dictionary: List[str], dictionary_length: int, entry: KEntry, explanations: List,
               original_classification, perturbation_tracking: List, next_iteration: List[KEntry], language: str,
               index: int):

        for i in range(dictionary_length):
            current_doc = entry.clone()
            current_doc.indices.add(i)
            current_doc.document[index] = i

            is_counterfactual, current_classification, src = self.check(dictionary, current_doc.document,
                                                                        original_classification, language)
            if is_counterfactual:
                explanations.append((current_doc.indices, current_classification, 1))
                perturbation_tracking.append(src)
            else:
                next_iteration.append(current_doc)

    def check(self, dictionary, document: List[int], original_classification, language: str) -> Tuple[bool, any, str]:
        src = format_code(self.proxy.perturb_positions(dictionary, document), language)
        print("checking\n",src)
        initial_output = self.proxy.classify(src)
        current_classification, unused = initial_output[0] if isinstance(initial_output, list) else \
            initial_output
        if current_classification != original_classification:
            return True, current_classification, src
        else:
            return False, None, src
