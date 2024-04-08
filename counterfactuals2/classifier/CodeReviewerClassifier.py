from transformers import RobertaForSequenceClassification, RobertaTokenizer

from common.diffGen import to_diff_code_reviewer
from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
import torch

from counterfactuals2.classifier.ReviewerModel import build_or_load_gen_model


class CodeReviewerClassifier(AbstractClassifier):
    model_path = "D:/A_Uni/A_MasterThesis/MMD/ClsModelFinetuned/checkpoints-21600-0.689"
    tokenizer_path = "microsoft/codereviewer"

    def __init__(self):
        print()
        config, model, tokenizer = build_or_load_gen_model(self.model_path, self.tokenizer_path)
        self.model = model
        self.tokenizer = tokenizer

    def classify(self, source_code: str, is_raw_input=False) -> (any, float):
        """Evaluates the input and returns a tuple with (result, confidence)"""
        source_code = source_code if is_raw_input else to_diff_code_reviewer(source_code)
        inputs = self.tokenizer(source_code, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(cls=True, input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                labels=None)
            prediction = torch.argmax(logits, dim=-1).cpu().numpy()
            return prediction, float(logits[0][prediction])
