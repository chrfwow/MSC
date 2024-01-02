from counterfactuals2.searchAlgorithms.GeneticSearchAlgorihm import GeneticSearchAlgorithm
from counterfactuals2.tokenizer.LineTokenizer import LineTokenizer
from counterfactuals2.classifier.PLBartClassifier import PLBartClassifier
from counterfactuals2.perturber.RemoveTokenPerturber import RemoveTokenPerturber
from counterfactuals2.counterfactual_search import CounterfactualSearch
from counterfactuals2.misc.language import Language

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
perturber = RemoveTokenPerturber()

print(classifier.classify(cpp_code))
print(classifier.classify("""#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    int a[2] = {1,2};
    return a[1];
}
""".strip()))

search_algorithm = GeneticSearchAlgorithm(tokenizer, classifier, perturber, language, iterations=3, gene_pool_size=10)

cf_search = CounterfactualSearch(language, tokenizer, search_algorithm)
counterfactuals = cf_search.search(cpp_code)

print("Found", len(counterfactuals), "counterfactuals")
for c in counterfactuals:
    print(c.to_string(tokenizer))
