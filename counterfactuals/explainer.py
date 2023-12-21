import time
from typing import List

from .GreedySearch import GreedySearch
from misc.code_formatter import format_code
from .counterfactual_search import BaseCounterfactualSearch
from .base_proxy import BasePerturbationProxy


class SequenceExplainer:
    def __init__(self, language: str, counterfactual_search: BaseCounterfactualSearch = None):
        self.language = language
        self.counterfactual_search = counterfactual_search if counterfactual_search is not None \
            else GreedySearch(BasePerturbationProxy())

    def explain(self, document):
        start = time.perf_counter()
        sequence, full_explanations, perturbation_tracking, full_dictionary, document_length = self.counterfactual_search.search(
            document)
        end = time.perf_counter()
        return SequenceExplanation(
            sequence,
            full_explanations,
            perturbation_tracking,
            self.language,
            execution_time=int(end - start),
            original_document=document,
            full_dictionary=full_dictionary,
            document_length=document_length,
        )


def strike(s):
    return '\u0336'.join(s) + '\u0336'


class SequenceExplanation:
    def __init__(
            self,
            document_sequence: List,
            explanations: List,
            perturbation_tracking: List,
            language: str,
            execution_time: int = 0,
            original_document=None,
            full_dictionary=None,
            document_length: int = 0
    ):
        self.language = language
        self.document_sequence = document_sequence
        self.explanations = explanations
        self.perturbation_tracking = perturbation_tracking
        self.execution_time = execution_time
        self.original_document = original_document
        if full_dictionary is None:
            self.full_dictionary = original_document.split(' ')
            self.document_length = len(self.full_dictionary)
        else:
            self.full_dictionary = full_dictionary
            self.document_length = document_length

    def has_explanations(self):
        return len(self.explanations) > 0

    # same as 'full' but without the positions
    def human_readable(self):
        return [
            list(map(lambda pos: self.full_dictionary[pos], explanation_list[0]))
            for explanation_list in self.explanations
        ]

    def set_original_document(self, original_document):
        self.original_document = original_document

    def full(self):
        return [
            (
                list(
                    map(
                        lambda pos: (pos, self.full_dictionary[pos]),
                        explanation_list[0],
                    )
                ),
                explanation_list[1],
            )
            for explanation_list in self.explanations
        ]

    # Returns a string representation as a list of explanations
    # Each explanation item is a tuple of document position and
    # item and document item at that position
    def __repr__(self):
        return str(self.full())

    def __str__(self):
        return str(self.human_readable())

    def print_removal_explanations(self):
        # extract only positional information
        positions = list(map(lambda x: sorted(list(map(lambda z: z[0], x))), map(lambda x: x[0], self.full())))
        for single_positions in positions:
            current_tokens = [*self.full_dictionary]
            for position in single_positions:
                current_tokens[position] = strike(current_tokens[position])
            print(" ".join(current_tokens))

    def print_explanations(self):
        positions = list(map(lambda x: list(map(lambda z: z[0], x)), map(lambda x: x[0], self.full())))

        original: List[str] = []
        for i in range(self.document_length):
            original.append(self.full_dictionary[i])
        original_doc = " ".join(original)

        original_doc = format_code(original_doc, self.language)
        original_doc_lines = original_doc.split("\n")

        i = 0
        for single_positions in positions:
            exp: List[str] = []
            for position in single_positions:
                exp.append(self.full_dictionary[position])

            exp_src = format_code(" ".join(exp), self.language)

            print("explanation", i, ":", )
            print_alternating(original_doc_lines, exp_src.split("\n"))
            print()
            i += 1


def print_alternating(original: List, explanation: List):
    for i in range(max(len(original), len(explanation))):
        if i < len(original):
            print("original   :", original[i])
        else:
            print("original   :")
        if i < len(explanation):
            print("explanation:", explanation[i])
        else:
            print("explanation:")
