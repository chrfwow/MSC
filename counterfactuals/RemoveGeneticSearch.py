import random
import string
import sys

from counterfactuals.base_proxy import BasePerturbationProxy
from counterfactuals.counterfactual_search import BaseCounterfactualSearch


def fitness(initial_score: float, current_score: float) -> float:
    return abs(initial_score - current_score)


def should_survive(current_fitness: float, min_fitness: float, delta_fitness: float) -> bool:
    scaled = (current_fitness - min_fitness) / delta_fitness
    return random.random() * scaled > .3  # todo fine tune


class RemoveGeneticSearch(BaseCounterfactualSearch):
    def __init__(self, proxy: BasePerturbationProxy, iterations: int = 10, gene_pool_size: int = 50,
                 kill_ratio: float = .4):
        self.proxy = proxy
        self.iterations = iterations
        self.gene_pool_size = gene_pool_size
        self.kill_ratio = kill_ratio

    def search(self, document, proxy: BasePerturbationProxy = None):
        if proxy is None:
            proxy = self.proxy

        random.seed(1)
        initial_output = proxy.classify(document)
        print(initial_output)

        initial_classification, initial_score = initial_output[0] if isinstance(initial_output, list) else \
            initial_output
        print(initial_classification, initial_score)

        sequence = proxy.document_to_perturbation_space(document)
        sequence_length = len(sequence)
        gene_pool = [Entry]

        explanations = []
        perturbation_tracking = []

        for i in range(self.gene_pool_size):  # add random start population
            entry = Entry("", 0, [])
            mutate(entry, sequence_length)
            gene_pool.append(entry)

        for iteration in range(self.iterations):
            print("iteration", iteration, "########################")
            print("starting evaluation...")

            for gene in gene_pool:
                gene_doc = proxy.perturb_positions(sequence, gene.candidates)
                gene_classification, gene_score = proxy.classify(gene_doc)

                current_fitness = fitness(initial_score, gene_score)
                print("gene_classification", gene_classification, "current_fitness", current_fitness, "gene_doc",
                      gene_doc)

                gene.classification = gene_classification

                if gene_classification != initial_classification:
                    explanations.append((gene.candidates, gene_classification, current_fitness))
                    perturbation_tracking.append(gene_doc)
                else:
                    gene.fitness = current_fitness

            # remove counterfactuals
            gene_pool = list(filter(lambda entry: entry.classification == initial_classification, gene_pool))
            if len(gene_pool) == 0:
                print("only counterfactuals left")
                break

            # sort highest fitness first
            gene_pool.sort(key=lambda entry: entry.fitness, reverse=True)

            size_before_massacre = len(gene_pool)
            kills = int(self.kill_ratio * size_before_massacre)

            best = gene_pool[0]
            best_killed = False
            for i in range(kills):
                rand = random.random()
                rand *= rand
                index = int(rand * len(gene_pool))
                if index == 0:
                    best_killed = True
                del gene_pool[index]

            if best_killed:
                gene_pool.insert(0, best)

            print("best: fitness", best.fitness, " candidates ", best.candidates, " best doc ",
                  proxy.perturb_positions(sequence, best.candidates))

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
                    rand_a = random.random()
                    rand_a *= rand_a
                    rand_a_index = int(rand_a * pool_size)
                    rand_b = random.random()
                    rand_b *= rand_b
                    rand_b_index = int(rand_b * pool_size)
                    parent_a = gene_pool[rand_a_index]
                    parent_b = gene_pool[rand_b_index]
                    gene_pool.append(make_offspring(parent_a, parent_b))

            # mutate existing genes
            mutations = 0
            for i in range(pool_size):
                if random.random() > .5:
                    mutate(gene_pool[i], sequence_length)
                    mutations += 1
            print(mutations, "mutations")

        print("expl", explanations)
        return sequence, explanations, perturbation_tracking


class Entry:
    classification = ""
    fitness = 0.0
    candidates = []

    def __init__(self, classification: string, fitness: float, candidates: [int]):
        self.classification = classification
        self.fitness = fitness
        self.candidates = candidates

    def clone(self):
        return Entry(self.classification, self.fitness, list(self.candidates))


def mutate(this: Entry, sequence_length: int):
    if len(this.candidates) == 0:
        this.candidates.append(int(sequence_length * random.random()))
        pass

    what = random.random()
    if what < .4:  # add candidate
        this.candidates.append(int(sequence_length * random.random()))
    elif what < .8:  # remove candidate
        del this.candidates[int(len(this.candidates) * random.random())]
    else:  # change candidate
        this.candidates[int(len(this.candidates) * random.random())] = int(sequence_length * random.random())


def make_offspring(a: Entry, b: Entry):
    return Entry(0, 0, list({*a.candidates, *b.candidates}))
