from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification

from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
import torch
import torch.nn as nn
import torch.nn.functional as functional


class CodeBertClassifier(AbstractClassifier):
    path = "D:/A_Uni/A_MasterThesis/CodeBertModel/models/CodeBERT/Vulnerability Detection/model/model.bin"
    model_type = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_type, truncation=True)

    def __init__(self, device):
        config = RobertaConfig.from_pretrained(self.model_type)
        config.num_labels = 1
        raw_model = RobertaForSequenceClassification.from_pretrained(self.model_type, config=config).to(device)
        self.model = Model(raw_model)
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.device = device

    def classify(self, source_code: str) -> (bool, float):
        """Evaluates the input and returns a tuple with (result, confidence)"""
        inputs = self.tokenizer(source_code, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            prob = functional.sigmoid(self.model(input_ids)).item()
            return round(prob) == 0, prob

    def get_max_tokens(self) -> int:
        return self.tokenizer.model_max_length

    def get_embeddings(self):
        """Returns the embeddings to be used by Layer Integrated Gradients"""
        return self.model.encoder.roberta.embeddings

    def get_logits(self, input_indices, attention_mask=None):
        """Returns the raw logits output of the model"""
        return self.model(input_ids=input_indices)

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


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = outputs
        return logits
