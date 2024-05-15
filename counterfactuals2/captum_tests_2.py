import torch

from counterfactuals2.classifier.PLBartClassifier import PLBartClassifier
from counterfactuals2.searchAlgorithms.LigSearch import LigSearch

cpp_code_easy = """
int main() {
    int* out = (int*) malloc(64 * sizeof(int));
    free(out);
    return out[3];
}
""".strip()

substring_search = """
int containsSubstring(char *input) {
    int stringLength = strlen(input);
    for(int i = 0; i < stringLength; i--){
        if(input[i] == 'W' && input[i + 1] == 'o')
            return 1;
    }
    return 0;
}
""".strip()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device", device)

classifier = PLBartClassifier(device)
# classifier = VulBERTa_MLP_Classifier(device)
#classifier = CodeBertClassifier(device)
#classifier = CodeT5Classifier(device)
#classifier = GraphCodeBertClassifier(device)

ligsearch = LigSearch(classifier, recompute_attributions_for_each_iteration=True)

counterfactuals = ligsearch.search(substring_search).counterfactuals

print("found", len(counterfactuals), "counterfactuals:")
for c in counterfactuals:
    print(c.to_string())
