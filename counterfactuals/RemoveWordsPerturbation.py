from typing import Tuple, List
from transformers import pipeline
from transformers import AutoTokenizer

from counterfactuals.base_proxy import BasePerturbationProxy
from common.diffGen import to_diff_hunk


class RemoveWordsPerturbation(BasePerturbationProxy):
    # tokenizer = AutoTokenizer.from_pretrained("razent/cotext-2-cc")
    # model = AutoModelForSequenceClassification.from_pretrained("razent/cotext-2-cc")

    t = AutoTokenizer.from_pretrained("uclanlp/plbart-c-cpp-defect-detection", use_fast=False)
    defect_pipeline = pipeline(model="uclanlp/plbart-c-cpp-defect-detection", tokenizer=t)

    # defect_pipeline = pipeline("text2text-generation", model="microsoft/codereviewer")

    def classify(self, document) -> Tuple[bool, float]:
        output = self.defect_pipeline([to_diff_hunk(document)])
        return output[0]['label'], output[0]['score']

    def document_to_perturbation_space(self, document: str) -> List[str]:
        return document.split()

    def perturb_positions(self, perturbation_space: List, positions: List[int]) -> List:
        perturbed_sequence = []
        for i in range(len(perturbation_space)):
            if i not in positions:
                perturbed_sequence.append(perturbation_space[i])
        return ' '.join(perturbed_sequence)
