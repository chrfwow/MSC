from typing import Tuple, List

import torch

from transformers import AutoTokenizer, PLBartForSequenceClassification
from transformers import pipeline

from counterfactuals.base_proxy import BasePerturbationProxy
from counterfactuals.code_formatter import format_code
from counterfactuals.diffGen import to_diff_hunk
import re


class ReplaceWordsPerturbationPlBart(BasePerturbationProxy):
    # tokenizer = AutoTokenizer.from_pretrained("razent/cotext-2-cc")
    # model = AutoModelForSequenceClassification.from_pretrained("razent/cotext-2-cc")

    defect_detection = "uclanlp/plbart-c-cpp-defect-detection"
    base = "uclanlp/plbart-base"
    finetuned = "mrm8488/codebert2codebert-finetuned-code-defect-detection"
    model_path = finetuned  # defect_detection

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PLBartForSequenceClassification.from_pretrained(model_path, problem_type="multi_label_classification")

    # t = AutoTokenizer.from_pretrained("mrm8488/codebert2codebert-finetuned-code-defect-detection", use_fast=False)
    # defect_pipeline = pipeline(model="mrm8488/codebert2codebert-finetuned-code-defect-detection", tokenizer=t)

    # defect_pipeline = pipeline("text2text-generation", model="microsoft/codereviewer")

    # add all keywords and such
    cpp_tokens: List[str] = ["", " ", "\n", "(", ")", "{", "}", "[", "]", ";", "\"", "<<", "<", ">>", ">",
                             "::", ".", ",", "+", "-", "*", "/", "+=", "-=", "*=", "/=", "!=", "==", "=",
                             "||", "|", "&&", "&", "~", "'", "->", "true", "false"]
    cpp_keywords: List[str] = ["do", "while", "for", "break", "return", "if", "else", "int", "bool", "double", "float",
                               "long", "const", "unsigned", "switch", "struct", "nullptr", "free", "malloc", "case",
                               "len", ]

    java_tokens: List[str] = ["", " ", "\n", "(", ")", "{", "}", "[", "]", ";", "\"", "<<", "<", ">>", ">",
                              "::", ".", ",", "+", "-", "*", "/", "+=", "-=", "*=", "/=", "!=", "==", "=",
                              "||", "|", "&&", "&", "'", "->", "true", "false"]
    java_keywords: List[str] = ["do", "while", "for", "break", "return", "if", "else", "int", "boolean", "double",
                                "float", "long", "final", "static", "switch", "null", "case", "String"]

    tokens = []
    keywords = []

    def __init__(self, lang: str):
        self.num_labels = len(self.model.config.id2label)
        self.model = PLBartForSequenceClassification.from_pretrained(self.model_path, num_labels=self.num_labels,
                                                                     problem_type="multi_label_classification")

        if lang == "cpp" or lang == "c++":
            self.tokens = self.cpp_tokens
            self.keywords = self.cpp_keywords
        elif lang == "java" or lang == "Java":
            self.tokens = self.java_tokens
            self.keywords = self.java_keywords
        else:
            print("unknown language", lang)
            return

        for i in range(10):
            self.tokens.append(str(i))

    def classify(self, document) -> Tuple[bool, float]:
        inputs = self.tokenizer(document, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
        labels = torch.sum(
            torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=self.num_labels), dim=1
        ).to(torch.float)
        loss = self.model(**inputs, labels=labels).loss

        return labels[0][0] >= .5, float(loss)

    def document_to_perturbation_space(self, document: str) -> (int, List[str]):
        """Returns a tuple containing the number of words in the document, and a list of all available words,
        including words not in the document"""

        escaped_dict: List[str] = []
        for token in self.tokens:
            if token != "" and token != " " and token != "\n":
                escaped_dict.append(re.escape(token))

        escaped_keywords: List[str] = []
        for word in self.keywords:
            escaped_keywords.append(word + "\\s+")

        delimiter = "|".join(escaped_dict) + "|" + "|".join(escaped_keywords)
        delimiter = "(\\s+|\\-?[0-9]*[\\.]?[0-9]+f?|\\-?[0-9]+l?|" + delimiter + ")"

        doc_parts = re.split(delimiter, document)
        doc_parts = [word.strip() for word in doc_parts]
        doc_parts = list(filter(lambda a: not re.match(r"\s+", a), doc_parts))
        doc_parts = list(filter(None, doc_parts))
        doc_parts_set = {*doc_parts}
        additions = [*self.tokens, *self.keywords]

        directory_without_doc_parts: List[str] = []

        for token in additions:
            if token not in doc_parts_set:
                directory_without_doc_parts.append(token)

        return len(doc_parts), [*doc_parts, *directory_without_doc_parts]

    def perturb_positions(self, perturbation_space: List[str], positions: List[int]) -> str:
        perturbed_sequence = []

        for i in positions:
            perturbed_sequence.append(perturbation_space[i])

        return ' '.join(perturbed_sequence)
