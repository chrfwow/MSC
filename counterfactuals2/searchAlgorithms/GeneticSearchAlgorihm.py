import random
import time
from typing import List, Set

from common.compileSourceCode import is_syntactically_correct
from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from counterfactuals2.misc.Counterfactual import Counterfactual
from counterfactuals2.misc.Entry import Entry
from counterfactuals2.misc.GeneticSearchParameters import GeneticSearchParameters
from counterfactuals2.misc.SearchParameters import SearchParameters
from counterfactuals2.misc.language import Language
from counterfactuals2.perturber.AbstractPerturber import AbstractPerturber
from counterfactuals2.searchAlgorithms.AbstractSearchAlgorithm import AbstractSearchAlgorithm
from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer
from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker


def no_duplicate(counterfactuals: List[Counterfactual], gene_doc: str) -> bool:
    for c in counterfactuals:
        if c.code == gene_doc:
            return False
    return True


class GeneticSearchAlgorithm(AbstractSearchAlgorithm):

    def __init__(self, tokenizer: AbstractTokenizer, evaluator: AbstractClassifier, perturber: AbstractPerturber,
                 iterations: int = 10, gene_pool_size: int = 50, kill_ratio: float = .3, allow_syntax_errors_in_counterfactuals: bool = True, verbose=False):
        super().__init__(tokenizer, evaluator, verbose)
        self.iterations = iterations
        self.gene_pool_size = gene_pool_size
        self.kill_ratio = kill_ratio
        self.perturber = perturber
        self.allow_syntax_errors_in_counterfactuals = allow_syntax_errors_in_counterfactuals
        self.verbose = verbose
        self.fitness_set = False

    def get_parameters(self) -> SearchParameters:
        return GeneticSearchParameters(self.iterations, self.gene_pool_size, self.kill_ratio, self.allow_syntax_errors_in_counterfactuals)

    def get_perturber(self) -> AbstractPerturber | None:
        return self.perturber

    def get_unmasker(self) -> AbstractUnmasker | None:
        return self.tokenizer.unmasker

    def perform_search(self, source_code: str, number_of_tokens_in_src: int, dictionary: List[str], original_class: any,
                       original_confidence: float, original_tokens: List[int]) -> List[Counterfactual]:
        random.seed(1)
        original_dictionary_length = len(dictionary)

        gene_pool: List[Entry] = []
        counterfactuals: List[Counterfactual] = []

        start_time = time.time()

        for i in range(self.gene_pool_size):  # add random start population
            entry = Entry("", 0, [*original_tokens])
            entry.changed_values.add(self.perturber.perturb_in_place(
                entry.document_indices, original_dictionary_length))
            entry.number_of_changes += 1
            if len(entry.document_indices) > 0:
                gene_pool.append(entry)

        for iteration in range(self.iterations):
            if self.verbose:
                print("iteration", iteration, "########################")
                print("gene pool size", len(gene_pool))
                print("already found", len(counterfactuals), "counterfactuals")
                print("starting evaluation...")

            to_remove: Set[Entry] = set()

            for gene in gene_pool:
                if gene.fitness_set:
                    continue
                try:
                    gene_doc = self.tokenizer.to_string_unmasked(
                        dictionary, gene.document_indices)
                    gene_classification, gene_score = self.classifier.classify(
                        gene_doc)
                except Exception as e:
                    to_remove.add(gene)
                    print(e)
                    continue
                if self.allow_syntax_errors_in_counterfactuals:
                    is_syntactically_correct_code = True
                else:
                    is_syntactically_correct_code = is_syntactically_correct(
                        gene_doc, Language.Cpp)
                current_fitness = fitness(is_syntactically_correct_code, gene, number_of_tokens_in_src,
                                          original_class, original_confidence, gene_classification, gene_score)

                gene.classification = gene_classification
                gene.fitness = current_fitness
                gene.fitness_set = True

                if gene_classification != original_class:
                    if is_syntactically_correct_code and no_duplicate(counterfactuals, gene_doc):
                        for i in gene.changed_values:
                            if type(i) != int:
                                print("aaa", i)
                            if i >= len(dictionary):
                                print("oje", i)
                        changed_lines = [
                            "" if i == AbstractTokenizer.EMPTY_TOKEN_INDEX else dictionary[i] for i in gene.changed_values]
                        counterfactuals.append(Counterfactual(gene_doc, current_fitness, start_time,
                                                              original_dictionary_length, gene.number_of_changes, len(gene.document_indices), changed_lines))
                    # todo really remove, or keep in gene pool to hopefully make it better?
                    to_remove.add(gene)

            # remove counterfactuals
            gene_pool = list(
                filter(lambda entry: entry not in to_remove, gene_pool))
            if len(gene_pool) == 0:
                if self.verbose:
                    print("only counterfactuals left")
                break

            kills = int(self.kill_ratio * len(gene_pool))

            best = get_best(gene_pool)
            for i in range(kills):
                index = roulette_worst(gene_pool)
                to_kill = gene_pool[index]
                if to_kill == best:
                    pass
                else:
                    del gene_pool[index]

            if self.verbose:
                print("best: fitness", best.fitness, " candidates ",
                      best.document_indices, " best doc")
                print(self.tokenizer.to_string(
                    dictionary, best.document_indices))

            pool_size = len(gene_pool)
            if self.verbose:
                print("killed ", kills, " genes because they were too bad")
            if pool_size == 0:
                if self.verbose:
                    print("no genes left")
                break

            missing_genes = self.gene_pool_size - pool_size
            if missing_genes > 0:
                if self.verbose:
                    print("making ", missing_genes,
                          " new offspring or mutations")

                offspring = 0
                mutations = 0

                # let the best genes reproduce or mutate
                for i in range(missing_genes):
                    if i % 2 == 0:
                        offspring += 1
                        parent_a = roulette_best(gene_pool)
                        parent_b = roulette_best(gene_pool)
                        child = make_offspring(parent_a, parent_b, dictionary)
                        if len(child.document_indices) > 0:
                            gene_pool.append(child)
                    else:
                        to_mutate = gene_pool[random.randint(
                            0, len(gene_pool) - 1)]
                        mutated = to_mutate.clone()
                        if len(mutated.document_indices) == 0:
                            continue
                        mutations += 1
                        mutated.changed_values.add(self.perturber.perturb_in_place(
                            mutated.document_indices, len(dictionary)))
                        mutated.number_of_changes += 1
                        gene_pool.append(mutated)

                if self.verbose:
                    print(mutations, "mutations")
                    print(offspring, "offspring")

        return counterfactuals


def make_offspring(a: Entry, b: Entry, dictionary):
    new_indices: List[int] = []

    if len(a.document_indices) < len(b.document_indices):
        shorter = a.document_indices
        longer = b.document_indices
    else:
        shorter = b.document_indices
        longer = a.document_indices

    shorter_length = len(shorter)
    pivot = int(shorter_length * random.random())
    for i in range(pivot):
        new_indices.append(shorter[i])

    for i in range(len(longer) - pivot):
        new_indices.append(longer[i + pivot])

    ret = Entry(0, 0, new_indices, max(a.number_of_changes,
                                       b.number_of_changes) + 1, {*a.changed_values, *b.changed_values})
    for r in ret.changed_values:
        if r >= len(dictionary):
            print("ahhh", r, "max", len(dictionary))
    return ret


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
    if initial_classification != current_classification:
        return 10000000
    score = initial_score - current_score
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
