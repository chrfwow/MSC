from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier


class VulBERTa_MLP_Classifier(AbstractClassifier):
    from transformers import pipeline
    pipe = pipeline("text-classification", model="claudios/VulBERTa-MLP-Devign", trust_remote_code=True, top_k=None)

    def classify(self, source_code: str) -> (any, float):
        """Evaluates the input and returns a tuple with (result, confidence)"""
        result = self.pipe(source_code)[0]
        a = result[0]
        b = result[1]
        score_a = a["score"]
        score_b = b["score"]
        if score_a > score_b:
            return a["label"], score_a
        else:
            return b["label"], score_b
