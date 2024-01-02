import random
import string
from typing import List, Set
from counterfactuals.ReplaceWordsPerturbation_plbart import ReplaceWordsPerturbationPlBart
from common.code_formatter import format_code
from common.compileSourceCode import is_syntactically_correct
from counterfactuals.counterfactual_search import BaseCounterfactualSearch


def should_survive(current_fitness: float, min_fitness: float, delta_fitness: float) -> bool:
    scaled = (current_fitness - min_fitness) / delta_fitness
    return random.random() * scaled > .3  # todo fine tune


forbidden_compiler_errors: List[str] = ["cannot combine with previous '", "expected '", "expected expression",
                                        "invalid parameter name", "type-id cannot have a name", "invalid '==' at",
                                        "a type specifier is required for all declarations", "extraneous closing brace",
                                        "invalid suffix on literal", "expected function body after function declarator",
                                        "expected unqualified-id", "missing terminating '"]


class GeneticSearch(BaseCounterfactualSearch):
    def __init__(self, language, iterations: int = 10, gene_pool_size: int = 50,
                 kill_ratio: float = .3):
        self.language = language
        self.proxy = ReplaceWordsPerturbationPlBart(language)
        self.iterations = iterations
        self.gene_pool_size = gene_pool_size
        self.kill_ratio = kill_ratio

    def search(self, document):
        print("searching for counterfactuals for\n", format_code(document, self.language))
        proxy = self.proxy

        random.seed(1)

        document_length, dictionary = proxy.document_to_perturbation_space(document)
        print("document_length", document_length, "dictionary", dictionary)

        dictionary_length = len(dictionary)

        gene_pool: List[Entry] = []
        explanations = []
        perturbation_tracking = []

        original_document_indices: List[int] = []
        for i in range(document_length):
            original_document_indices.append(i)

        initial_output = proxy.classify(proxy.perturb_positions(dictionary, original_document_indices))
        print("initial_output", initial_output)

        initial_classification, initial_score = initial_output[0] if isinstance(initial_output, list) else \
            initial_output
        print("initial_classification", initial_classification, initial_score)

        for i in range(self.gene_pool_size):  # add random start population
            entry = Entry("", 0, [*original_document_indices])
            mutate(entry, dictionary_length)
            gene_pool.append(entry)

        for iteration in range(self.iterations):
            print("iteration", iteration, "########################")
            print("gene pool size", len(gene_pool))
            print("starting evaluation...")

            to_remove: Set[Entry] = set()

            for gene in gene_pool:
                gene_doc = format_code(proxy.perturb_positions(dictionary, gene.document_indices), self.language)
                gene_classification, gene_score = proxy.classify(gene_doc)
                is_syntactically_correct_code = is_syntactically_correct(gene_doc, self.language)
                current_fitness = fitness(is_syntactically_correct_code, gene, document_length,
                                          initial_classification, initial_score, gene_classification, gene_score)
                # print("gene_classification", gene_classification, "current_fitness", current_fitness, "gene_doc",
                #      gene_doc)

                gene.classification = gene_classification
                gene.fitness = current_fitness

                if gene_classification != initial_classification:
                    # print("is_syntactically_correct", is_syntactically_correct, ":", gene_doc)
                    if is_syntactically_correct_code:
                        explanations.append((gene.document_indices, gene_classification, current_fitness))
                        perturbation_tracking.append(gene_doc)
                        to_remove.add(gene)  # todo really remove, or keep in gene pool to hopefully make it better?

            # remove counterfactuals
            gene_pool = list(filter(lambda entry: entry not in to_remove, gene_pool))
            if len(gene_pool) == 0:
                print("only counterfactuals left")
                break

            # sort highest fitness last
            # gene_pool.sort(key=lambda entry: entry.fitness, reverse=False)

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
            print(format_code(proxy.perturb_positions(dictionary, best.document_indices), self.language))

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
                    mutate(gene_pool[i], dictionary_length)
                    mutations += 1
            print(mutations, "mutations")

        print("expl", explanations)
        return document, explanations, perturbation_tracking, dictionary, document_length


class Entry:
    classification:any = ""
    fitness = 0.0
    document_indices: List[int] = []

    def __init__(self, classification: any, fitness: float, document_indices: [int]):
        self.classification = classification
        self.fitness = fitness
        self.document_indices = document_indices

    def clone(self):
        return Entry(self.classification, self.fitness, list(self.document_indices))


def mutate(this: Entry, dictionary_length: int):
    what = random.random()
    if what < .3:  # add candidate
        this.document_indices.insert(int(len(this.document_indices) * random.random()),
                                     int(dictionary_length * random.random()))
    elif what < .6:  # remove candidate
        del this.document_indices[int(len(this.document_indices) * random.random())]
    else:  # change candidate
        this.document_indices[int(len(this.document_indices) * random.random())] = int(
            dictionary_length * random.random())


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
        total_fitness += 1 / entry.fitness

    rand = random.random() * total_fitness
    offset = 0.0
    for i in range(len(entries)):
        offset += 1 / entries[i].fitness
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
