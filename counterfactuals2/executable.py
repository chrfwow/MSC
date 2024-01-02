from counterfactuals2.searchAlgorithms.GeneticSearchAlgorihm import GeneticSearchAlgorithm
from counterfactuals2.searchAlgorithms.KExpExhaustiveMaskedSearch import KExpExhaustiveMaskedSearch
from counterfactuals2.tokenizer.LineTokenizer import LineTokenizer
from counterfactuals2.classifier.PLBartClassifier import PLBartClassifier
from counterfactuals2.perturber.RemoveTokenPerturber import RemoveTokenPerturber
from counterfactuals2.counterfactual_search import CounterfactualSearch
from counterfactuals2.misc.language import Language
from counterfactuals2.tokenizer.RegexTokenizer import RegexTokenizer
from counterfactuals2.unmasker.CodeBertUnmasker import CodeBertUnmasker

cpp_code = """
#include <iostream>

int main() {
    std::cout << 'as' << std::endl;
    return 0;
}
""".strip()
classifier = PLBartClassifier()

print(classifier.classify(cpp_code))
print(classifier.classify("""
#include <iostream>

int main() {
    std::cout << "a" << std::endl;
    return 0;
}
""".strip()))

language = Language.Cpp
unmasker = CodeBertUnmasker()
tokenizer = RegexTokenizer(language, unmasker)

perturber = RemoveTokenPerturber()

search_algorithm = KExpExhaustiveMaskedSearch(1, unmasker, tokenizer, classifier, language)
# search_algorithm = GeneticSearchAlgorithm(tokenizer, classifier, perturber, language, iterations=3, gene_pool_size=10)

cf_search = CounterfactualSearch(language, tokenizer, search_algorithm)
counterfactuals = cf_search.search(cpp_code)

print("Found", len(counterfactuals), "counterfactuals")
for c in counterfactuals:
    print(c.to_string())
