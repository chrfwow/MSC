from typing import List

from counterfactuals2.misc.language import Language
from common.code_formatter import format_code


class AbstractTokenizer:
    mask = "<<mask>>"  # todo replace with real maks

    def __init__(self, language: Language):
        self.language = language

    def tokenize(self, source_code: str) -> (int, List[str]):
        """Returns a tuple containing the number of words in the document, and a list of all available words,
        including words not in the document"""
        raise NotImplementedError

    def to_string(self, dictionary: List[str], tokens: List[int]) -> str:
        perturbed_sequence = []

        for i in tokens:
            perturbed_sequence.append(dictionary[i])

        return format_code(' '.join(perturbed_sequence), self.language)

    def to_string_with_mask(self, dictionary: List[str], tokens: List[int], replace_with_mask: int | List[int]) -> str:
        perturbed_sequence = []
        if isinstance(replace_with_mask, int):
            for i in tokens:
                if i == replace_with_mask:
                    perturbed_sequence.append(self.mask)
                else:
                    perturbed_sequence.append(dictionary[i])
        else:
            for i in tokens:
                if i in replace_with_mask:
                    perturbed_sequence.append(self.mask)
                else:
                    perturbed_sequence.append(dictionary[i])
        return format_code(' '.join(perturbed_sequence), self.language)
