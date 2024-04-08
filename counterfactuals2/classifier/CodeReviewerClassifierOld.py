from transformers import RobertaForSequenceClassification, RobertaTokenizer

from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
import torch


class CodeReviewerClassifierOld(AbstractClassifier):
    path = "microsoft/codereviewer"
    model = RobertaForSequenceClassification.from_pretrained(path, problem_type="multi_label_classification")
    num_labels = len(model.config.id2label)
    tokenizer = RobertaTokenizer.from_pretrained(path)

    def classify(self, source_code: str) -> (any, float):
        """Evaluates the input and returns a tuple with (result, confidence)"""
        inputs = self.tokenizer(source_code, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            clazz = logits.argmax().item()
            predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
            print(logits)
            return predicted_class_ids, 1
