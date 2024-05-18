from counterfactuals2.misc.SearchParameters import SearchParameters


class GreedySearchParameters(SearchParameters):
    def __init__(self, iterations: int, max_age: int, max_survivors: int):
        super().__init__(iterations)
        self.max_age = max_age
        self.max_survivors = max_survivors
