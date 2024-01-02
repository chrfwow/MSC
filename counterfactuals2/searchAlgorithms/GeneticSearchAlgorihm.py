import random
from typing import List, Set

from common.compileSourceCode import is_syntactically_correct
from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from counterfactuals2.misc.Counterfactual import Counterfactual
from counterfactuals2.misc.Entry import Entry
from counterfactuals2.misc.language import Language
from counterfactuals2.perturber.AbstractPerturber import AbstractPerturber
from counterfactuals2.searchAlgorithms.AbstractSearchAlgorithm import AbstractSearchAlgorithm
from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer


class GeneticSearchAlgorithm(AbstractSearchAlgorithm):

    def __init__(self, tokenizer: AbstractTokenizer, evaluator: AbstractClassifier, perturber: AbstractPerturber,
                 language: Language, iterations: int = 10, gene_pool_size: int = 50, kill_ratio: float = .3):
        super().__init__(tokenizer, evaluator, language)
        self.iterations = iterations
        self.gene_pool_size = gene_pool_size
        self.kill_ratio = kill_ratio
        self.perturber = perturber

    def perform_search(self, source_code: str, number_of_tokens_in_src: int, dictionary: List[str], original_class: any,
                       original_confidence: float, original_tokens: List[int]) -> List[Counterfactual]:
        random.seed(1)
        dictionary_length = len(dictionary)

        gene_pool: List[Entry] = []
        counterfactuals: List[Counterfactual] = []

        for i in range(self.gene_pool_size):  # add random start population
            entry = Entry("", 0, [*original_tokens])
            self.perturber.perturb_in_place(entry.document_indices, dictionary_length)
            gene_pool.append(entry)

        for iteration in range(self.iterations):
            print("iteration", iteration, "########################")
            print("gene pool size", len(gene_pool))
            print("starting evaluation...")

            to_remove: Set[Entry] = set()

            for gene in gene_pool:
                gene_doc = self.tokenizer.to_string(dictionary, gene.document_indices)
                gene_classification, gene_score = self.classifier.classify(gene_doc)
                is_syntactically_correct_code = is_syntactically_correct(gene_doc, self.language)
                current_fitness = fitness(is_syntactically_correct_code, gene, number_of_tokens_in_src,
                                          original_class, original_confidence, gene_classification, gene_score)

                gene.classification = gene_classification
                gene.fitness = current_fitness

                if gene_classification != original_class:
                    if is_syntactically_correct_code:
                        counterfactuals.append(Counterfactual(gene_doc, current_fitness))
                        to_remove.add(gene)  # todo really remove, or keep in gene pool to hopefully make it better?

            # remove counterfactuals
            gene_pool = list(filter(lambda entry: entry not in to_remove, gene_pool))
            if len(gene_pool) == 0:
                print("only counterfactuals left")
                break

            kills = int(self.kill_ratio * len(gene_pool))

            best = get_best(gene_pool)
            best_killed = False
            for i in range(kills):
                index = roulette_worst(gene_pool)
                to_kill = gene_pool[index]
                if to_kill == best:
                    best_killed = True
                del gene_pool[index]

            if best_killed:
                gene_pool.insert(len(gene_pool), best)

            print("best: fitness", best.fitness, " candidates ", best.document_indices, " best doc")
            print(self.tokenizer.to_string(dictionary, best.document_indices))

            pool_size = len(gene_pool)
            print("killed ", kills, " genes because they were too bad")
            if pool_size == 0:
                print("no genes left")
                break

            missing_genes = self.gene_pool_size - pool_size
            if missing_genes > 0:
                print("making ", missing_genes, " new offspring")

                # let the best genes reproduce
                for i in range(missing_genes):
                    parent_a = roulette_best(gene_pool)
                    parent_b = roulette_best(gene_pool)
                    gene_pool.append(make_offspring(parent_a, parent_b))

            # mutate existing genes
            mutations = 0
            for i in range(pool_size):
                if random.random() > .5:
                    self.perturber.perturb_in_place(gene_pool[i].document_indices, dictionary_length)
                    mutations += 1
            print(mutations, "mutations")

        return counterfactuals


def make_offspring(a: Entry, b: Entry):
    new_indices: List[int] = []

    if len(a.document_indices) < len(b.document_indices):
        shorter = a.document_indices
        longer = b.document_indices
    else:
        shorter = b.document_indices
        longer = a.document_indices

    shorter_length = len(shorter)
    for i in range(shorter_length):
        if random.random() < .5:
            new_indices.append(shorter[i])
        else:
            new_indices.append(longer[i])

    for i in range(len(longer) - shorter_length):
        new_indices.append(longer[i + shorter_length])

    return Entry(0, 0, new_indices)


# def fitness(entry: Entry, document_length: int, initial_score: float, current_score: float) -> float:
#     delta_length = abs(len(entry.document_indices) - document_length)
#     relative_delta = delta_length / float(document_length)
#     punishment = 1 - relative_delta
#     if punishment < 0:
#         punishment = 0
#     return abs(initial_score - current_score) * punishment


def fitness(is_syntactically_correct_code: bool, entry: Entry, document_length: int,
            initial_classification: any,
            initial_score: float,
            current_classification: any, current_score: float) -> float:
    if initial_classification == current_classification:
        score = current_score - initial_score
    else:
        if current_score < 0.0001:
            score = 1000000
        else:
            score = 1.0 / current_score
    if not is_syntactically_correct_code:
        score *= 0.25
    delta_length = abs(len(entry.document_indices) - document_length)
    relative_delta = delta_length / float(document_length)
    punishment = 1.0 - relative_delta
    if punishment < 0:
        punishment = 0
    punishment = (punishment + 1) * .5
    return punishment * score


def roulette_best(entries: List[Entry]) -> Entry:
    total_fitness = 0.0
    for entry in entries:
        total_fitness += entry.fitness

    rand = random.random() * total_fitness
    offset = 0.0
    for entry in entries:
        offset += entry.fitness
        if rand < offset:
            return entry
    return entries[len(entries) - 1]


def roulette_worst(entries: List[Entry]) -> int:
    total_fitness = 0.0
    for entry in entries:
        fit = entry.fitness
        if fit < 0.001:
            fit = 0.001
        total_fitness += 1 / fit

    rand = random.random() * total_fitness
    offset = 0.0
    for i in range(len(entries)):
        fit = entries[i].fitness
        if fit < 0.001:
            fit = 0.001
        offset += 1 / fit
        if rand < offset:
            return i
    return len(entries) - 1


def get_best(entries: List[Entry]) -> Entry:
    best = entries[0]
    best_fitness = entries[0].fitness
    for entry in entries:
        if entry.fitness > best_fitness:
            best = entry
            best_fitness = best.fitness
    return best
