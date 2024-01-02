from typing import List

from common.code_formatter import format_code
from counterfactuals2.misc.language import Language
from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker


class AbstractTokenizer:
    def __init__(self, language: Language, unmasker: AbstractUnmasker | None = None):
        self.language = language
        self.unmasker = unmasker
        if unmasker is None:
            self.mask = None
        else:
            self.mask = unmasker.get_mask()

    def tokenize(self, source_code: str) -> (int, List[str]):
        """Returns a tuple containing the number of words in the document, and a list of all available words,
        including words not in the document"""
        raise NotImplementedError

    def to_string(self, dictionary: List[str], tokens: List[int]) -> str:
        perturbed_sequence = []

        for i in tokens:
            if i == AbstractUnmasker.MASK_INDEX:
                perturbed_sequence.append(self.mask)
            else:
                perturbed_sequence.append(dictionary[i])

        return format_code(' '.join(perturbed_sequence), self.language)

    def to_string_unmasked(self, dictionary: List[str], tokens: List[int], replace_with_mask: int) -> str:
        if self.mask is None:
            raise Exception("mask or unmasker is None")
        perturbed_sequence = []

        for i in tokens:
            if i == replace_with_mask or i == AbstractUnmasker.MASK_INDEX:
                code = format_code(self.to_string(dictionary, tokens), self.language, self.mask)
                replacement = self.unmasker.get_mask_replacement(code)
                perturbed_sequence.append(replacement)
            else:
                perturbed_sequence.append(dictionary[i])

        return format_code(' '.join(perturbed_sequence), self.language)
