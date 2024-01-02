class Counterfactual:
    def __init__(self, code: str, score: float):
        self.code = code
        self.score = score

    def to_string(self):
        return "score " + str(self.score) + ": " + self.code
