from typing import Tuple, List

from transformers import AutoTokenizer
from transformers import pipeline

from counterfactuals.base_proxy import BasePerturbationProxy
from counterfactuals.code_formatter import format_code
from counterfactuals.diffGen import to_diff_hunk
import re


class ReplaceWordsPerturbation(BasePerturbationProxy):
    # tokenizer = AutoTokenizer.from_pretrained("razent/cotext-2-cc")
    # model = AutoModelForSequenceClassification.from_pretrained("razent/cotext-2-cc")

    t = AutoTokenizer.from_pretrained("uclanlp/plbart-c-cpp-defect-detection", use_fast=False)
    defect_pipeline = pipeline(model="uclanlp/plbart-c-cpp-defect-detection", tokenizer=t)

    # t = AutoTokenizer.from_pretrained("mrm8488/codebert2codebert-finetuned-code-defect-detection", use_fast=False)
    # defect_pipeline = pipeline(model="mrm8488/codebert2codebert-finetuned-code-defect-detection", tokenizer=t)

    # defect_pipeline = pipeline("text2text-generation", model="microsoft/codereviewer")

    # add all keywords and such
    additional_dictionary: List[str] = ["", " ", "\n", "do", "while", "for", "(", ")", "{", "}", "[", "]", ";",
                                        "break;", "return", "<<", "<", ">", "::", ".", ",", "+", "-", "*", "/", "+=",
                                        "-=", "*=", "/=", "if", "else", "!=", "==", "|", "||", "&", "&&", "~", "int",
                                        "bool", "double", "float", "long", "[]", "const", "unsigned", "switch",
                                        "struct", "nullptr", "free", "malloc", "case", "->"]

    def classify(self, document) -> Tuple[bool, float]:
        # output = self.defect_pipeline([to_diff_hunk(document)])
        # document = to_diff_hunk(document)
        formatted = format_code(document)
        output = self.defect_pipeline([formatted])
        return output[0]['label'], output[0]['score']

    def document_to_perturbation_space(self, document: str) -> (int, List[str]):
        """Returns a tuple containing the number of words in the document, and a list of all available words,
        including words not in the document"""

        escaped_dict: List[str] = []
        for word in self.additional_dictionary:
            if word != "" and word != " " and word != "\n":
                escaped_dict.append(re.escape(word))
        delimiter = "|".join(escaped_dict)
        delimiter = "(\\s+|" + delimiter + ")"

        doc_parts = re.split(delimiter, document, flags=re.IGNORECASE)
        doc_parts = [word.strip() for word in doc_parts]
        doc_parts = list(filter(lambda a: not re.match(r"\s+", a), doc_parts))
        doc_parts = list(filter(None, doc_parts))
        doc_parts_set = {*doc_parts}
        additions = [*self.additional_dictionary]

        directory_without_doc_parts: List[str] = []

        for word in additions:
            if word not in doc_parts_set:
                directory_without_doc_parts.append(word)

        return len(doc_parts), [*doc_parts, *directory_without_doc_parts]

    def perturb_positions(self, perturbation_space: List[str], positions: List[int]) -> str:
        perturbed_sequence = []

        for i in positions:
            perturbed_sequence.append(perturbation_space[i])

        return ' '.join(perturbed_sequence)
