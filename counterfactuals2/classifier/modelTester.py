import shutil
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from counterfactuals2.misc.language import Language
from counterfactuals2.perturber.RemoveTokenPerturber import RemoveTokenPerturber
from counterfactuals2.searchAlgorithms.KExpExhaustiveSearch import KExpExhaustiveSearch
from counterfactuals2.tokenizer.LineTokenizer import LineTokenizer
from counterfactuals2.unmasker.NoOpUnmasker import NoOpUnmasker

cpp_code = """
#include <iostream>

int main() {
    std::cout << "a" << std::endl;
    int* a = malloc(sizeof(int) * 64);
    return 0;
}
""".strip()


class SomeClassifier(AbstractClassifier):
    def __init__(self, path):
        self.path = path
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def classify(self, source_code: str) -> (bool, float):
        inputs = self.tokenizer(source_code, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            clazz = logits.argmax().item()
            return int(clazz) == 0, float(logits[0][clazz])


paths: List[str] = ["mrm8488/codebert2codebert-finetuned-code-defect-detection",
                    "mcanoglu/microsoft-codebert-base-finetuned-defect-detection",
                    "mcanoglu/Salesforce-codet5p-770m-finetuned-defect-detection",
                    "mcanoglu/deepseek-ai-deepseek-coder-1.3b-base-finetuned-defect-detection",
                    "mcanoglu/bigcode-starcoderbase-1b-finetuned-defect-detection",
                    "mcanoglu/deepseek-ai-deepseek-coder-1.3b-base-finetuned-defect-detection",
                    "starmage520/Coderbert_finetuned_detect_vulnerability_on_MSR",
                    "Zaib/Vulnerability-detection",
                    "neuralsentry/vulnerabilityDetection-StarEncoder-Devign",
                    ]

working_models: List[str] = []

language = Language.Cpp
unmasker = NoOpUnmasker()
tokenizer = LineTokenizer(language, unmasker)
perturber = RemoveTokenPerturber()

for path in paths:
    try:
        classifier = SomeClassifier(path)
        search_algorithm = KExpExhaustiveSearch(1, unmasker, tokenizer, classifier, perturber, language)
        counterfactuals = search_algorithm.search(cpp_code)
        if len(counterfactuals) > 0:
            working_models.append(path)
            print("found", len(counterfactuals), "counterfactuals with", path)
    except:
        print("exception when trying model", path)
    path_on_hdd = "C:/Users/Christian/.cache/huggingface/hub/models--" + path.replace("/", "--")
    try:
        print("deleting folder", path_on_hdd)
        shutil.rmtree(path_on_hdd)
    except:
        print("could not delete folder", path_on_hdd)

print("#" * 10)
print("#" * 10)
print("#" * 10)
print("working models")
for working_model in working_models:
    print(working_model)

# working models
# mcanoglu/Salesforce-codet5p-770m-finetuned-defect-detection
# mcanoglu/deepseek-ai-deepseek-coder-1.3b-base-finetuned-defect-detection
# mcanoglu/bigcode-starcoderbase-1b-finetuned-defect-detection
