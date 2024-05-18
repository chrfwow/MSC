from typing import List


class Entry:
    classification: any
    fitness: float
    document_indices: List[int]

    def __init__(self, classification: any, fitness: float, document_indices: [int], number_of_changes: int = 0, changed_values: List[int] = []):
        self.classification = classification
        self.fitness = fitness
        self.document_indices = document_indices
        self.number_of_changes = number_of_changes
        self.changed_values = changed_values

    def clone(self):
        return Entry(self.classification, self.fitness, list(self.document_indices), self.number_of_changes, list(self.changed_values))
