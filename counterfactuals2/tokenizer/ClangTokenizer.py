from counterfactuals2.clangInit import init_clang

init_clang()
import clang.cindex

idx = clang.cindex.Index.create()

from typing import List
from counterfactuals2.tokenizer.AbstractTokenizer import AbstractTokenizer


class ClangTokenizer(AbstractTokenizer):
    def get_joining_string(self) -> str:
        return " "

    def tokenize(self, source_code: str) -> (int, List[int], List[str]):
        tu = idx.parse('tmp.cpp', args=['-std=c++11'], unsaved_files=[('tmp.cpp', source_code)], options=0)
        tokens = [t for t in tu.get_tokens(extent=tu.cursor.extent)]

        d = dict()  # [str] = index
        indices = []
        current_index = 0

        for t in tokens:
            r = t.spelling
            if r in d.keys():
                indices.append(d[r])
            else:
                d[r] = current_index
                indices.append(current_index)
                current_index += 1

        if "free" not in d.keys():
            d["free"] = current_index
            current_index += 1
        if "malloc" not in d.keys():
            d["malloc"] = current_index

        return len(indices), indices, list(d.keys())
