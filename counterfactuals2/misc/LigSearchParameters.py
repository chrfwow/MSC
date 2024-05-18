from counterfactuals2.misc.SearchParameters import SearchParameters


class LigSearchParameters(SearchParameters):
    def __init__(self, iterations: int, steps_per_iteration: int, recompute_attributions_for_each_iteration: bool):
        super().__init__(iterations)
        self.steps_per_iteration = steps_per_iteration
        self.recompute_attributions_for_each_iteration = recompute_attributions_for_each_iteration