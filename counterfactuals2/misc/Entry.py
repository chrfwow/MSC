from typing import List


class Entry:
    classification: any
    fitness: float
    document_indices: List[int]

    def __init__(self, classification: any, fitness: float, document_indices: [int]):
        self.classification = classification
        self.fitness = fitness
        self.document_indices = document_indices

    def clone(self):
        return Entry(self.classification, self.fitness, list(self.document_indices))
