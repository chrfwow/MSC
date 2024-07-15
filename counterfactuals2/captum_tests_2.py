from counterfactuals2.classifier.PLBartClassifier import PLBartClassifier

from counterfactuals2.clangInit import init_clang

init_clang()

from counterfactuals2.unmasker.CodeBertUnmasker import CodeBertUnmasker
from counterfactuals2.searchAlgorithms.KExpExhaustiveSearch import KExpExhaustiveSearch
from counterfactuals2.perturber.MaskedPerturber import MaskedPerturber
import torch
from counterfactuals2.tokenizer.ClangTokenizer import ClangTokenizer
from counterfactuals2.misc.DatasetLoader import load_code_x_glue
from classifier.CodeBertClassifier import CodeBertClassifier

vulnerable_source_codes = load_code_x_glue(skip=6 + 3 - 1, keep=1)[0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

unmasker = CodeBertUnmasker(device)
tokenizer = ClangTokenizer(unmasker)
classifier = PLBartClassifier(device)
perturber = MaskedPerturber()
keexp = KExpExhaustiveSearch(2, unmasker, tokenizer, classifier, perturber, False)
print("search started")
print(keexp.search(vulnerable_source_codes))
