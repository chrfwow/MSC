import clang.cindex

from counterfactuals2.classifier.VulBERTa_MLP_Classifier import VulBERTa_MLP_Classifier

clang.cindex.Config.set_library_file('D:/Programme/LLVM/bin/libclang.dll')
index = clang.cindex.Index.create()

from counterfactuals2.classifier.CodeReviewerClassifier import CodeReviewerClassifier
from counterfactuals2.searchAlgorithms.GeneticSearchAlgorihm import GeneticSearchAlgorithm
from counterfactuals2.searchAlgorithms.KExpExhaustiveSearch import KExpExhaustiveSearch
from counterfactuals2.tokenizer.LineTokenizer import LineTokenizer
from counterfactuals2.classifier.PLBartClassifier import PLBartClassifier
from counterfactuals2.perturber.RemoveTokenPerturber import RemoveTokenPerturber
from counterfactuals2.counterfactual_search import CounterfactualSearch
from counterfactuals2.misc.language import Language
from counterfactuals2.tokenizer.RegexTokenizer import RegexTokenizer
from counterfactuals2.unmasker.CodeBertUnmasker import CodeBertUnmasker
from counterfactuals2.unmasker.NoOpUnmasker import NoOpUnmasker

cpp_code = """
#include <iostream>

int main() {
    std::cout << "a" << std::endl;
    int* a = malloc(sizeof(int) * 64);
    return 0;
}
""".strip()
# classifier = CodeReviewerClassifier()
# classifier = PLBartClassifier()
classifier = VulBERTa_MLP_Classifier()

print(classifier.classify(cpp_code))
print(classifier.classify("""
int main() {
    std::cout << "a" << std::endl;
    return 0;
}
""".strip()))

language = Language.Cpp
# unmasker = CodeBertUnmasker()
# tokenizer = RegexTokenizer(language, unmasker)

unmasker = NoOpUnmasker()
tokenizer = LineTokenizer(language, unmasker)

perturber = RemoveTokenPerturber()

search_algorithm = KExpExhaustiveSearch(1, unmasker, tokenizer, classifier, perturber, language)
# search_algorithm = GeneticSearchAlgorithm(tokenizer, classifier, perturber, language, iterations=30, gene_pool_size=10)

cf_search = CounterfactualSearch(language, tokenizer, search_algorithm)
counterfactuals = cf_search.search(cpp_code)

print("Found", len(counterfactuals), "counterfactuals")
for c in counterfactuals:
    print(c.to_string())
