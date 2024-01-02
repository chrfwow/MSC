class AbstractClassifier:
    def classify(self, source_code: str) -> (any, float):
        """Evaluates the input and returns a tuple with (result, confidence)"""
        raise NotImplementedError
