from transformers import AutoModelForSequenceClassification, AutoTokenizer

from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
import torch


class CodeT5Classifier(AbstractClassifier):
    path = "mcanoglu/Salesforce-codet5p-770m-finetuned-defect-detection"
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path, truncation=True)

    def __init__(self, device):
        self.device = device
        self.model = self.model.to(device)

    def classify(self, source_code: str) -> (bool, float):
        """Evaluates the input and returns a tuple with (result, confidence). Result is True iff source_code is assumed to be ok"""
        inputs = self.tokenizer(source_code, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            clazz = logits.argmax().item()
            return int(clazz) == 0, float(logits[0][clazz])

    def get_max_tokens(self) -> int:
        return self.tokenizer.model_max_length

    def get_embeddings(self):
        """Returns the embeddings to be used by Layer Integrated Gradients"""
        return self.model._modules["transformer"].shared

    def get_logits(self, input_indices, attention_mask):
        """Returns the raw logits output of the model"""
        return self.model(input_ids=input_indices, attention_mask=attention_mask).logits

    def prepare_for_lig(self, device):
        """Prepares the model for LIG by moving it to the given device next to other measures"""
        self.model.to(device)
        self.model.eval()
        self.model.zero_grad()

    def token_string_to_id(self, token_str: str) -> int:
        """Gives the token id for a given token string"""
        return self.tokenizer._convert_token_to_id(token_str)

    def token_id_to_string(self, token_id: int) -> str:
        """Gives the token string for a given token id"""
        return self.tokenizer._convert_id_to_token(token_id)

    def get_begin_of_string_token_id(self) -> int:
        """Gives the token id for the begin of string token"""
        return self.tokenizer.bos_token_id

    def get_end_of_string_token_id(self) -> int:
        """Gives the token id for the end of string token"""
        return self.tokenizer.eos_token_id

    def get_padding_token_id(self) -> int:
        """Gives the token id for the padding token"""
        return self.tokenizer.pad_token_id

    def tokenize(self, input: str) -> dict:
        """Uses the model spcific tokenizer to tokenize the input string"""
        return self.tokenizer(input, truncation=True)
