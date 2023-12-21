from counterfactuals.explainer import SequenceExplainer

import torch

from counterfactuals.kExponentialSearch import KExponentialSearch

# tokenizer = AutoTokenizer.from_pretrained('mrm8488/codebert-base-finetuned-detect-insecure-code')
# model = AutoModelForSequenceClassification.from_pretrained('mrm8488/codebert-base-finetuned-detect-insecure-code')
#
# src = "int main(){\n   int a = 1;\n   printf(\"%d\", a);\n   return 0;\n}"
# print("src", "\n" + src)
# inputs = tokenizer(src, return_tensors="pt", truncation=True, padding='max_length')
# print("inputs", inputs)
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# print("labels", labels)
# outputs = model(**inputs, labels=labels)
# print("outputs", outputs)
# loss = outputs.loss
# logits = outputs.logits
#
# print("loss", loss)
# print("logits", logits)
# print()
# print("1: insecure code, 0: secure code")
# print(np.argmax(logits.detach().numpy()))

pa = """
bool isEveryOddCharacter(string &content, char c) {
    // thread safety
    if(content == null) {
        for(int i = 1; i < content.length(); i+=2) {
            if(content[i] != c) {
                return false;
            }
        }
    }
    return true;
} 
""".strip()
pb = """
int generatePNG(vector<unsigned char> image, string outputName)
{
	const string outputPath = directory + outputName;
	if (lodepng::encode(outputPath, image, width, height) != 0) {
		cout << "Error encoding PNG" << endl;
		return 1;
	}
	return 0;
}
""".strip()
pc = """
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    int a = 0;
    return a;
}
""".strip()
pjava = """
public class Main{
public static void main(String[] args){
    break System.out.println("hello");
}}
""".strip()

p1 = pjava
language = "java"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used", device)
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead", device)

# sequenceExplainer = SequenceExplainer(GreedySearch(RemoveWordsPerturbation()))
# sequenceExplainer = SequenceExplainer(language, GeneticSearch(language=language, iterations=15, gene_pool_size=90))
sequenceExplainer = SequenceExplainer(language, KExponentialSearch(language=language, k=1))
explanations = sequenceExplainer.explain(p1)

# explanations.print_removal_explanations()
explanations.print_explanations()
