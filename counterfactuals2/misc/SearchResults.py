from typing import List
from DatasetLoader import skipped

ids_of_inputs = dict()  # [int] = str
inputs_of_ids = dict()  # [str] = int
current_id = skipped


class SearchResult:
    def __init__(
            self,
            input: str,
            input_token_length: int,
            counterfactuals: List,
            search_algorithm,
            classifier,
            perturber,
            tokenizer,
            unmasker,
            search_duration: float,
            parameters,
            truncated: bool
    ):
        global current_id
        if input in inputs_of_ids:
            self.input_id = inputs_of_ids[input]
        else:
            inputs_of_ids[input] = current_id
            ids_of_inputs[current_id] = input
            self.input_id = current_id
            current_id += 1

        self.input_token_length = input_token_length
        self.parameters = parameters
        self.search_duration = search_duration
        self.counterfactuals = counterfactuals
        self.search_algorithm = search_algorithm.__class__.__name__
        self.classifier = classifier.__class__.__name__
        self.truncated = truncated
        if unmasker is None:
            self.unmasker = "None"
        else:
            self.unmasker = unmasker.__class__.__name__

        if perturber is None:
            self.perturber = "None"
        else:
            self.perturber = perturber.__class__.__name__

        if tokenizer is None:
            self.tokenizer = "None"
        else:
            self.tokenizer = tokenizer.__class__.__name__

    def to_string(self) -> str:
        res = "Search result for " + \
              self.search_algorithm + \
              " with classifier " + \
              self.classifier + ", perturber " + \
              self.perturber + ", tokenizer " + \
              self.tokenizer + ", unmasker " + \
              self.unmasker + " took " + \
              str(self.search_duration) + "sec and produced " + \
              str(len(self.counterfactuals)) + \
              " counterfactuals. Truncated = " + \
              str(self.truncated) + \
              ". Input string:\n" + ids_of_inputs[self.input_id]
        if len(self.counterfactuals) == 0:
            return res
        res += "\nThese are:\n"
        for i in range(len(self.counterfactuals)):
            res += "#" + str(i) + ":\n" + self.counterfactuals[i].to_string() + "\n"
        return res


class SearchError(SearchResult):
    def __init__(
            self,
            input: str,
            input_token_length: int,
            cause: Exception,
            search_algorithm,
            classifier,
            perturber,
            tokenizer,
            unmasker,
            search_duration: float,
            parameters,
            truncated: bool
    ):
        super().__init__(input, input_token_length, [], search_algorithm, classifier, perturber, tokenizer, unmasker, search_duration, parameters, truncated)
        self.cause = str(cause)

    def to_string(self) -> str:
        res = "Search for " + \
              self.search_algorithm + \
              " with classifier " + \
              self.classifier + ", perturber " + \
              self.perturber + ", tokenizer " + \
              self.tokenizer + ", unmasker " + \
              self.unmasker + ", truncated " + \
              str(self.truncated) + " took " + \
              str(self.search_duration) + "sec and produced en error: " + \
              str(self.cause) + " for input " + \
              str(ids_of_inputs[self.input_id])
        return res


class InvalidClassificationResult(SearchResult):
    def __init__(
            self,
            input: str,
            input_token_length: int,
            classification,
            search_algorithm,
            classifier,
            perturber,
            tokenizer,
            unmasker,
            search_duration: float,
            parameters,
            truncated: bool
    ):
        super().__init__(input, input_token_length, [], search_algorithm, classifier, perturber, tokenizer, unmasker, search_duration, parameters, truncated)
        self.classification = classification

    def to_string(self) -> str:
        res = "Search for " + \
              self.search_algorithm + \
              " with classifier " + \
              self.classifier + ", perturber " + \
              self.perturber + ", tokenizer " + \
              self.tokenizer + ", unmasker " + \
              self.unmasker + ", truncated " + \
              str(self.truncated) + " took " + \
              str(self.search_duration) + "sec produced a classification of " + str(self.classification) + " for the input" + ids_of_inputs[self.input_id]
        return res


class NotApplicable:
    pass
