from typing import List

from common.code_formatter import format_code
from counterfactuals2.misc.language import Language
from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker


class AbstractTokenizer:
    EMPTY_TOKEN_INDEX = -2

    def __init__(self, unmasker: AbstractUnmasker | None = None):
        self.mask = None
        self.unmasker = None
        self.set_unmasker(unmasker)

    def set_unmasker(self, unmasker: AbstractUnmasker | None):
        self.unmasker = unmasker
        if unmasker is None:
            self.mask = None
        else:
            self.mask = unmasker.get_mask()

    def get_joining_string(self) -> str:
        """Returns the string used to join the list of tokens when converting tokens to strings"""
        raise NotImplementedError

    def tokenize(self, source_code: str) -> (int, List[int], List[str]):
        """Returns a tuple containing the number of tokens in the document, a list contacting the indices of the document, and a list of all available words,
        including words not in the document"""
        raise NotImplementedError

    def to_string(self, dictionary: List[str], tokens: List[int]) -> str:
        perturbed_sequence = []

        for i in tokens:
            if i == AbstractUnmasker.MASK_INDEX:
                perturbed_sequence.append(self.mask)
            elif i == self.EMPTY_TOKEN_INDEX:
                pass
            else:
                perturbed_sequence.append(dictionary[i])

        return format_code(self.get_joining_string().join(perturbed_sequence), Language.Cpp)

    def to_string_unmasked(self, dictionary: List[str], tokens: List[int], replace_with_mask: int = -1) -> str:
        perturbed_sequence = []

        i = 0
        for token in tokens:
            if token == self.EMPTY_TOKEN_INDEX:
                pass
            elif token == replace_with_mask or token == AbstractUnmasker.MASK_INDEX:
                tokens[token] = AbstractUnmasker.MASK_INDEX
                code = format_code(self.to_string(dictionary, tokens), Language.Cpp, self.mask)
                tokens[i] = self.unmasker.get_mask_replacement(token, code, dictionary)
                perturbed_sequence.append(dictionary[token])
            else:
                perturbed_sequence.append(dictionary[token])
            i += 1

        return format_code(self.get_joining_string().join(perturbed_sequence), Language.Cpp)
