from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier

error = False
no_error = True

to_classify = """
public class Main{
    public void main(String[] args){
        int i = 0;
        System.null.println(i);
        String s = "someString";
        int b = 23;
        System.out.println(s / b);
    }
}
"""


class ManualClassifier(AbstractClassifier):

    def classify(self, source_code: str) -> (bool, float):
        hits = 0
        confidence = 0.92
        if "System.null.println(i);" in source_code:
            confidence -= 0.3
            hits += 1
        if "System.out.println(s / b);" in source_code:
            confidence -= 0.2
            hits += 1

        if hits == 0:
            return no_error, 0.8
        return error, confidence
