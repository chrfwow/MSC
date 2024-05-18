import datetime
import time
from typing import List

import torch
import tokenizers
import transformers

from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from counterfactuals2.perturber.MaskedPerturber import MaskedPerturber
from counterfactuals2.perturber.MutationPerturber import MutationPerturber
from counterfactuals2.perturber.RemoveTokenPerturber import RemoveTokenPerturber
from counterfactuals2.searchAlgorithms.GeneticSearchAlgorihm import GeneticSearchAlgorithm
from counterfactuals2.searchAlgorithms.GreedySearchAlgorithm import GreedySearchAlgorithm
from counterfactuals2.searchAlgorithms.KExpExhaustiveSearch import KExpExhaustiveSearch
from counterfactuals2.tokenizer.LineTokenizer import LineTokenizer
from counterfactuals2.tokenizer.RegexTokenizer import RegexTokenizer
from counterfactuals2.unmasker.CodeBertUnmasker import CodeBertUnmasker
from counterfactuals2.unmasker.NoOpUnmasker import NoOpUnmasker
import json

# vulnerable_source_codes: List[str] = load_code_x_glue(50)
vulnerable_source_codes: List[str] = [
    """
void helper_slbie(CPUPPCState *env, target_ulong addr){
    PowerPCCPU *cpu = ppc_env_get_cpu(env);
    ppc_slb_t *slb;
    slb = slb_lookup(cpu, addr);
    if (!slb) return;
    if (slb->esid & SLB_ESID_V) {
        slb->esid &= ~SLB_ESID_V;
        tlb_flush(CPU(cpu), 1);
    }
}
""".strip()
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_results_to_json_file(results: List, total_duration: float):
    content = json.dumps([*results, {"duration_sec": total_duration}], default=lambda o: o.__dict__)
    file_name = "json_dump_" + transformers.__version__ + "_" + datetime.datetime.now().strftime("%Y_%B_%d__%H_%M_%S") + ".json"
    print("writing content to", file_name)
    with open(file_name, "w") as file:
        file.write(content)


def evaluate(classifiers: List[AbstractClassifier], srcs: List[str], results: List, verbose: bool = False):
    tokenizers = [LineTokenizer(), RegexTokenizer()]
    perturbers = [MaskedPerturber(), MutationPerturber(), RemoveTokenPerturber()]

    noop_unmasker = NoOpUnmasker()
    code_bert_unmasker = CodeBertUnmasker()
    for classifier in classifiers:
        for tokenizer in tokenizers:
            for perturber in perturbers:
                if type(perturber) == MaskedPerturber:
                    unmasker = code_bert_unmasker
                else:
                    unmasker = noop_unmasker
                tokenizer.set_unmasker(unmasker)

                search_algos = [
                    # GreedySearchAlgorithm(25, unmasker, tokenizer, classifier, perturber, verbose=verbose),
                    GeneticSearchAlgorithm(tokenizer, classifier, perturber, 20, 30, verbose=verbose),
                    # KExpExhaustiveSearch(2, unmasker, tokenizer, classifier, perturber, verbose=verbose)
                ]

                for search_algorithm in search_algos:
                    for src in srcs:
                        results.append(search_algorithm.search(src))


def ligsearch(classifiers: List[AbstractClassifier], srcs: List[str], results: List, verbose: bool = False):
    from counterfactuals2.searchAlgorithms.LigSearch import LigSearch

    for classifier in classifiers:
        for src in srcs:
            ligsearch = LigSearch(classifier, recompute_attributions_for_each_iteration=True, verbose=verbose)
            results.append(ligsearch.search(src))
            ligsearch = LigSearch(classifier, recompute_attributions_for_each_iteration=False, verbose=verbose)
            results.append(ligsearch.search(src))


def evaluate_transformers_v_4_17_0(srcs: List[str], verbose: bool = False):
    print("evaluate_transformers_v_4_17_0")
    from counterfactuals2.classifier.CodeBertClassifier import CodeBertClassifier
    from counterfactuals2.classifier.PLBartClassifier import PLBartClassifier
    classifiers = [CodeBertClassifier(device), PLBartClassifier(device)]
    results = []
    start_time = time.time()
    evaluate(classifiers, srcs, results, verbose)
    ligsearch(classifiers, srcs, results, verbose)
    for r in results:
        print(r.to_string())
    end_time = time.time()
    print("search took " + str(end_time - start_time))
    write_results_to_json_file(results, end_time - start_time)


def evaluate_transformers_v_4_37_0(srcs: List[str], verbose: bool = False):
    print("evaluate_transformers_v_4_37_0")
    from counterfactuals2.classifier.VulBERTa_MLP_Classifier import VulBERTa_MLP_Classifier
    from counterfactuals2.classifier.CodeT5Classifier import CodeT5Classifier
    vulberta = VulBERTa_MLP_Classifier(device)
    classifiers = [vulberta, CodeT5Classifier(device)]
    results = []
    start_time = time.time()
    evaluate(classifiers, srcs, results, verbose)
    ligsearch([vulberta], srcs, results, verbose)
    for r in results:
        print(r.to_string())
    end_time = time.time()
    print("search took " + str(end_time - start_time))
    write_results_to_json_file(results, end_time - start_time)


if __name__ == '__main__':
    verbose = False
    if transformers.__version__ == "4.17.0":
        evaluate_transformers_v_4_17_0(vulnerable_source_codes, verbose)
    else:
        evaluate_transformers_v_4_37_0(vulnerable_source_codes, verbose)
