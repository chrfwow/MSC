import time


class Counterfactual:
    def __init__(self, code: str, score: float, start_time: float, number_of_tokens_in_input: int, number_of_changes: int, number_of_tokens: int, changed_tokens: [str]):
        self.code = code
        self.score = score
        self.duration = time.time() - start_time
        self.number_of_changes = number_of_changes
        self.number_of_tokens_in_input = number_of_tokens_in_input
        self.number_of_tokens = number_of_tokens
        self.changed_tokens = set(changed_tokens)

    def get_time_per_token(self) -> float:
        return self.number_of_tokens_in_input / self.duration

    def get_percentage_of_changed_tokens(self) -> float:
        return self.number_of_changes / self.number_of_tokens_in_input

    def get_relative_length_to_input(self) -> float:
        return self.number_of_tokens / self.number_of_tokens_in_input

    def to_string(self):
        return "score " + str(self.score) + ":\n" + self.code

    def __str__(self):
        return self.to_string()
