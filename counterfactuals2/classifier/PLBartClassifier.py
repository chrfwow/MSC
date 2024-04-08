from transformers import PLBartForSequenceClassification, AutoTokenizer

from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
import torch


class PLBartClassifier(AbstractClassifier):
    path = "uclanlp/plbart-c-cpp-defect-detection"
    model = PLBartForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    def classify(self, source_code: str) -> (any, float):
        """Evaluates the input and returns a tuple with (result, confidence)"""
        inputs = self.tokenizer(source_code, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            clazz = logits.argmax().item()
            return clazz, float(logits[0][clazz])
