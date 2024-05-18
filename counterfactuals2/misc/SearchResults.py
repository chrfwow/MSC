from typing import List


class SearchResult:
    def __init__(
            self,
            input: str,
            counterfactuals: List,
            search_algorithm,
            classifier,
            perturber,
            tokenizer,
            unmasker,
            search_duration: float,
            parameters
    ):
        self.paramters = parameters
        self.input = input
        self.search_duration = search_duration
        self.counterfactuals = counterfactuals
        self.search_algorithm = search_algorithm.__class__.__name__
        self.classifier = classifier.__class__.__name__
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
              " counterfactuals. Input string:\n" + self.input
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
            cause: Exception,
            search_algorithm,
            classifier,
            perturber,
            tokenizer,
            unmasker,
            search_duration: float,
            parameters
    ):
        super().__init__(input, [], search_algorithm, classifier, perturber, tokenizer, unmasker, search_duration, parameters)
        self.cause = str(cause)

    def to_string(self) -> str:
        res = "Search for " + \
              self.search_algorithm + \
              " with classifier " + \
              self.classifier + ", perturber " + \
              self.perturber + ", tokenizer " + \
              self.tokenizer + ", unmasker " + \
              self.unmasker + " took " + \
              str(self.search_duration) + "sec and produced en error: " + \
              str(self.cause)
        return res


class InvalidClassificationResult(SearchResult):
    def __init__(
            self,
            input: str,
            classification,
            search_algorithm,
            classifier,
            perturber,
            tokenizer,
            unmasker,
            search_duration: float,
            parameters
    ):
        super().__init__(input, [], search_algorithm, classifier, perturber, tokenizer, unmasker, search_duration, parameters)
        self.classification = classification

    def to_string(self) -> str:
        res = "Search for " + \
              self.search_algorithm + \
              " with classifier " + \
              self.classifier + ", perturber " + \
              self.perturber + ", tokenizer " + \
              self.tokenizer + ", unmasker " + \
              self.unmasker + " took " + \
              str(self.search_duration) + "sec produced a classification of " + str(self.classification) + " for the input"
        return res
