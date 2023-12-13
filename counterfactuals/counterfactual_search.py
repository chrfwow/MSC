from .base_proxy import BasePerturbationProxy


class BaseCounterfactualSearch:
    def search(self, document, proxy: BasePerturbationProxy):
        raise NotImplementedError
