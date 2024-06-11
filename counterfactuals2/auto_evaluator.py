import json
from unmasker.NoOpUnmasker import NoOpUnmasker
from unmasker.CodeBertUnmasker import CodeBertUnmasker
from tokenizer.LineTokenizer import LineTokenizer
from searchAlgorithms.KExpExhaustiveSearch import KExpExhaustiveSearch
from searchAlgorithms.GreedySearchAlgorithm import GreedySearchAlgorithm
from searchAlgorithms.GeneticSearchAlgorihm import GeneticSearchAlgorithm
from perturber.RemoveTokenPerturber import RemoveTokenPerturber
from perturber.MutationPerturber import MutationPerturber
from perturber.MaskedPerturber import MaskedPerturber
from classifier.AbstractClassifier import AbstractClassifier
import transformers
import tokenizers
import torch
from typing import List
import time
import datetime
from tokenizer.ClangTokenizer import ClangTokenizer
from misc.SearchResults import ids_of_inputs
from misc.DatasetLoader import load_code_x_glue
import threading

from clangInit import init_clang

init_clang()


vulnerable_source_codes: List[str] = load_code_x_glue(keep=2)
# vulnerable_source_codes: List[str] = [
#     """
# void helper_slbie(CPUPPCState *env, target_ulong addr){
#     PowerPCCPU *cpu = ppc_env_get_cpu(env);
#     ppc_slb_t *slb;
#     slb = slb_lookup(cpu, addr);
#     if (!slb) return;
#     if (slb->esid & SLB_ESID_V) {
#         slb->esid &= ~SLB_ESID_V;
#         tlb_flush(CPU(cpu), 1);
#     }
# }
# """.strip()
# ]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

is_running = True
finished = False


def write_results_to_json_file(results: List, total_duration: float):
    content = json.dumps({"duration_sec": total_duration, "ids_of_input": ids_of_inputs,
                         "results": results}, default=lambda o: o.__dict__)
    file_name = "json_dump_" + transformers.__version__ + "_" + \
        datetime.datetime.now().strftime("%Y_%B_%d__%H_%M_%S") + ".json"
    print("writing content to", file_name)
    with open(file_name, "w") as file:
        file.write(content)


def evaluate(classifiers: List[AbstractClassifier], src: str, results: List, verbose: bool = False):
    tokenizers = [
        LineTokenizer(),
        ClangTokenizer()
    ]
    perturbers = [
        MaskedPerturber(),
        MutationPerturber(),
        RemoveTokenPerturber()
    ]

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
                    GreedySearchAlgorithm(
                        25, unmasker, tokenizer, classifier, perturber, verbose=verbose),
                    GeneticSearchAlgorithm(
                        tokenizer, classifier, perturber, 10, 30, verbose=verbose),
                    KExpExhaustiveSearch(
                        2, unmasker, tokenizer, classifier, perturber, verbose=verbose)
                ]

                for search_algorithm in search_algos:
                    print(
                        "starting with",
                        search_algorithm.__class__.__name__,
                        unmasker.__class__.__name__,
                        tokenizer.__class__.__name__,
                        classifier.__class__.__name__,
                        perturber.__class__.__name__
                    )

                    results.append(search_algorithm.search(src))


def ligsearch(classifiers: List[AbstractClassifier], src: str, results: List, verbose: bool = False):
    from counterfactuals2.searchAlgorithms.LigSearch import LigSearch

    for classifier in classifiers:
        print("starting with LigSearch with " + classifier.__class__.__name__)
        ligsearch = LigSearch(
            classifier, recompute_attributions_for_each_iteration=True, verbose=verbose)
        results.append(ligsearch.search(src))
        ligsearch = LigSearch(
            classifier, recompute_attributions_for_each_iteration=False, verbose=verbose)
        results.append(ligsearch.search(src))


def evaluate_transformers_v_4_17_0(srcs: List[str], verbose: bool = False):
    print("evaluate_transformers_v_4_17_0")
    from counterfactuals2.classifier.CodeBertClassifier import CodeBertClassifier
    from counterfactuals2.classifier.PLBartClassifier import PLBartClassifier
    classifiers = [
        CodeBertClassifier(device),
        PLBartClassifier(device)
    ]
    run_eval(classifiers, classifiers, srcs, verbose)


def evaluate_transformers_v_4_37_0(srcs: List[str], verbose: bool = False):
    print("evaluate_transformers_v_4_37_0")
    from counterfactuals2.classifier.VulBERTa_MLP_Classifier import VulBERTa_MLP_Classifier
    from counterfactuals2.classifier.CodeT5Classifier import CodeT5Classifier
    vulberta = VulBERTa_MLP_Classifier(device)
    classifiers = [
        vulberta,
        CodeT5Classifier(device)
    ]
    run_eval(classifiers, [vulberta], srcs, verbose)


def run_eval(classifiers: List[AbstractClassifier], lig_classifiers: List[AbstractClassifier], srcs: List[str], verbose: bool):
    results = []
    start_time = time.time()
    i = 1
    for src in srcs:
        print("evaluating source code snippet #", i)
        i += 1
        start = time.time()
        evaluate(classifiers, src, results, verbose)
        ligsearch(lig_classifiers, src, results, verbose)
        end = time.time()
        print("Source code #" + str(i) + " took " + str(end - start) + " sec")
        if not is_running:
            print("search aborted")
            break
    end_time = time.time()
    print("search took " + str(end_time - start_time))
    write_results_to_json_file(results, end_time - start_time)


def start():
    global finished
    verbose = False
    if transformers.__version__ == "4.17.0":
        evaluate_transformers_v_4_17_0(vulnerable_source_codes, verbose)
    else:
        evaluate_transformers_v_4_37_0(vulnerable_source_codes, verbose)
    finished = True


if __name__ == '__main__':
    is_running = True

    print("starting thread")

    thread = threading.Thread(target=start)
    thread.start()

    try:
        while not finished:
            time.sleep(1)
        print("Finished")
    except KeyboardInterrupt:
        is_running = False
        print("terminating, pls wait, this could take a lot of time")
