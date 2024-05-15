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
            search_duration: float
    ):
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
            res += "#" + str(i) + ":\n" + self.counterfactuals[i].to_string()
        return res
