import shutil
from typing import List
import pip
import transformers
import torch
import time

from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from counterfactuals2.misc import SearchResults
from counterfactuals2.perturber.RemoveTokenPerturber import RemoveTokenPerturber
from counterfactuals2.searchAlgorithms.GreedySearchAlgorithm import GreedySearchAlgorithm
from counterfactuals2.tokenizer.LineTokenizer import LineTokenizer
from counterfactuals2.unmasker.NoOpUnmasker import NoOpUnmasker

vulnerable_source_codes: List[str] = [
    """
int main() {
    int a = 34;
    free(a);
    return a;
}
    """.strip()
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(classifiers: List[AbstractClassifier], srcs: List[str], results: List):
    tokenizer = LineTokenizer()
    unmasker = NoOpUnmasker()
    perturber = RemoveTokenPerturber()

    for classifier in classifiers:
        search_algorithm = GreedySearchAlgorithm(100, unmasker, tokenizer, classifier, perturber)
        for src in srcs:
            results.append(search_algorithm.search(src))


def ligsearch(classifiers: List[AbstractClassifier], srcs: List[str], results: List):
    from counterfactuals2.searchAlgorithms.LigSearch import LigSearch

    for classifier in classifiers:
        for src in srcs:
            ligsearch = LigSearch(classifier, recompute_attributions_for_each_iteration=True)
            results.append(ligsearch.search(src))


def evaluate_transformers_v_4_17_0(srcs: List[str]):
    print("evaluate_transformers_v_4_17_0")
    from counterfactuals2.classifier.CodeBertClassifier import CodeBertClassifier
    from counterfactuals2.classifier.PLBartClassifier import PLBartClassifier
    classifiers = [CodeBertClassifier(device), PLBartClassifier(device)]
    results = []
    evaluate(classifiers, srcs, results)
    ligsearch(classifiers, srcs, results)
    for r in results:
        print(r.to_string())


def evaluate_transformers_v_4_37_0(srcs: List[str]):
    print("evaluate_transformers_v_4_37_0")
    from counterfactuals2.classifier.VulBERTa_MLP_Classifier import VulBERTa_MLP_Classifier
    from counterfactuals2.classifier.CodeT5Classifier import CodeT5Classifier
    vulberta = VulBERTa_MLP_Classifier(device)
    classifiers = [vulberta, CodeT5Classifier(device)]
    results = []
    evaluate(classifiers, srcs, results)
    ligsearch([vulberta], srcs, results)
    for r in results:
        print(r.to_string())


if __name__ == '__main__':
    if transformers.__version__ == "4.17.0":
        evaluate_transformers_v_4_17_0(vulnerable_source_codes)
    else:
        evaluate_transformers_v_4_37_0(vulnerable_source_codes)
