from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from transformers import pipeline


class VulBERTa_MLP_Classifier(AbstractClassifier):
    def __init__(self, device):
        self.pipe = pipeline("text-classification", model="claudios/VulBERTa-MLP-Devign", trust_remote_code=True, top_k=None, device=device, truncation=True)

    def classify(self, source_code: str) -> (bool, float):
        """Evaluates the input and returns a tuple with (result, confidence). Result is True iff source_code is assumed to be ok"""
        result = self.pipe(source_code)[0]
        a = result[0]
        b = result[1]
        score_a = a["score"]
        score_b = b["score"]
        if score_a > score_b:
            return True, score_a
        else:
            return False, score_b

    def get_max_tokens(self) -> int:
        return self.pipe.tokenizer.model_max_length

    def get_embeddings(self):
        """Returns the embeddings to be used by Layer Integrated Gradients"""
        return self.pipe.model._modules["roberta"].embeddings

    def get_logits(self, input_indices, attention_mask):
        """Returns the raw logits output of the model"""
        return self.pipe.model(input_indices).logits

    def prepare_for_lig(self, device):
        """Prepares the model for LIG by moving it to the given device next to other measures"""
        self.pipe.model.to(device)
        self.pipe.model.eval()
        self.pipe.model.zero_grad()

    def token_string_to_id(self, token_str: str) -> int:
        """Gives the token id for a given token string"""
        return self.pipe.tokenizer._convert_token_to_id_with_added_voc(token_str)

    def token_id_to_string(self, token_id: int) -> str:
        """Gives the token string for a given token id"""
        return self.pipe.tokenizer._convert_id_to_token(token_id)

    def get_begin_of_string_token_id(self) -> int:
        """Gives the token id for the begin of string token"""
        return 0

    def get_end_of_string_token_id(self) -> int:
        """Gives the token id for the end of string token"""
        return 2

    def get_padding_token_id(self) -> int:
        """Gives the token id for the padding token"""
        return 1

    def tokenize(self, input: str) -> dict:
        """Uses the model spcific tokenizer to tokenize the input string"""
        return self.pipe.tokenizer(input, truncation=True)
