import jellyfish

from counterfactuals2.clangInit import init_clang

init_clang()

from counterfactuals2.misc.DatasetLoader import load_code_x_glue
from counterfactuals2.tokenizer.ClangTokenizer import ClangTokenizer
from counterfactuals2.tokenizer.LineTokenizer import LineTokenizer

src_snippets = []

tokenizers = [
    LineTokenizer(),
    ClangTokenizer()
]

tokenizers_dict = dict()

for t in tokenizers:
    tokenizers_dict[t.__class__.__name__] = t


def find(raw_results, number_of_files):
    global src_snippets
    if len(src_snippets) == 0:
        src_snippets = load_code_x_glue(keep=number_of_files * 4)
        print("init src snippets with", number_of_files * 4, "snippets")

    lens = dict()
    for raw in raw_results:
        if raw["tokenizer"] == "NotApplicable":
            continue
        l = int(raw["input_token_length"])
        lens[raw["tokenizer"]] = l

    candidates = set()
    for i, src in enumerate(src_snippets):
        fits = True
        for name, length in lens.items():
            actual_length = tokenizers_dict[name].tokenize(src)[0]
            if actual_length != length:
                fits = False
                break
        if fits:
            candidates.add(i)

    if len(candidates) == 0:
        return raw_results[0]["input_id"]
    elif len(candidates) == 1:
        return list(candidates)[0]
    else:
        cfs = []
        for raw in raw_results:
            c = raw["counterfactuals"]
            if len(c) == 0:
                continue
            for cf in c:
                cfs.append(cf["code"])

        best_index = -1
        best_score = -1
        for i in candidates:
            src = src_snippets[i]
            str_sim = 0
            for cf in cfs:
                sim = jellyfish.jaro_similarity(src, cf)
                str_sim += sim

            if best_index < 0:
                best_index = i
                best_score = str_sim
            elif str_sim > best_score:
                best_index = i
                best_score = str_sim
            elif str_sim == best_score and str_sim > 0:
                print("collision")

        if best_index < 0:
            return raw_results[0]["input_id"]

        return best_index
