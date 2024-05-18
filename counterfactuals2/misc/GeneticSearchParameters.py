from counterfactuals2.misc.SearchParameters import SearchParameters


class GeneticSearchParameters(SearchParameters):
    def __init__(self, iterations: int, gene_pool_size: int, kill_ratio: float, allow_syntax_errors_in_counterfactuals: bool):
        super().__init__(iterations)
        self.gene_pool_size = gene_pool_size
        self.kill_ratio = kill_ratio
        self.allow_syntax_errors_in_counterfactuals = allow_syntax_errors_in_counterfactuals
