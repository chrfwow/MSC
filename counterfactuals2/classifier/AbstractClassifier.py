class AbstractClassifier:
    def classify(self, source_code: str) -> (bool, float):
        """Evaluates the input and returns a tuple with (result, confidence). Result is True iff source_code is assumed to be ok"""
        raise NotImplementedError

    def get_max_tokens(self) -> int:
        raise NotImplementedError

    def get_embeddings(self):
        """Returns the embeddings to be used by Layer Integrated Gradients"""
        raise NotImplementedError

    def get_logits(self, input_indices, attention_mask):
        """Returns the raw logits output of the model"""
        raise NotImplementedError

    def prepare_for_lig(self, device):
        """Prepares the model for LIG by moving it to the given device next to other measures"""
        raise NotImplementedError

    def token_string_to_id(self, token_str: str) -> int:
        """Gives the token id for a given token string"""
        raise NotImplementedError

    def token_id_to_string(self, token_id: int) -> str:
        """Gives the token string for a given token id"""
        raise NotImplementedError

    def get_begin_of_string_token_id(self) -> int:
        """Gives the token id for the begin of string token"""
        raise NotImplementedError

    def get_end_of_string_token_id(self) -> int:
        """Gives the token id for the end of string token"""
        raise NotImplementedError

    def get_padding_token_id(self) -> int:
        """Gives the token id for the padding token"""
        raise NotImplementedError

    def tokenize(self, input: str) -> dict:
        """Uses the model spcific tokenizer to tokenize the input string"""
        raise NotImplementedError
