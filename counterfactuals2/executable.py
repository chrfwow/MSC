from counterfactuals2.GeneticSearchAlgorihm import GeneticSearchAlgorithm
from counterfactuals2.LineTokenizer import LineTokenizer
from counterfactuals2.MutationPerturber import MutationPerturber
from counterfactuals2.PLBartClassifier import PLBartClassifier
from counterfactuals2.RegexTokenizer import RegexTokenizer
from counterfactuals2.RemoveTokenPerturber import RemoveTokenPerturber
from counterfactuals2.counterfactual_search import CounterfactualSearch
from counterfactuals2.language import Language

cpp_code = """
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    int a[2] ={1,2};
    return a["as"];
}
""".strip()

language = Language.Cpp
tokenizer = LineTokenizer(language)
classifier = PLBartClassifier()

print(classifier.classify(cpp_code))
print(classifier.classify("""#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    int a[2] = {1,2};
    return a[1];
}
""".strip()))

perturber = RemoveTokenPerturber()
search_algorithm = GeneticSearchAlgorithm(tokenizer, classifier, perturber, language, iterations=3, gene_pool_size=10)

cf_seach = CounterfactualSearch(language, tokenizer, search_algorithm)
counterfactuals = cf_seach.search(cpp_code)

for c in counterfactuals:
    print()
