import json
from typing import List

skipped = 0


def load_code_x_glue(skip: int = 0, keep: int = -1):
    global skipped
    skipped = skip
    file = open("D:/A_Uni/A_MasterThesis/CodeXGlue/function.json")
    content = file.read()
    dataset = json.loads(content)

    indices_file = open("D:/A_Uni/A_MasterThesis/CodeXGlue/valid.txt")
    indices_content = indices_file.read().split("\n")
    indices = set()
    for l in indices_content:
        indices.add(int(l))

    vulnerable: List[str] = []
    if keep <= 0:
        keep = len(dataset)
    for i, entry in enumerate(dataset):
        if i not in indices:
            continue
        if entry["target"] == 1:
            if skip > 0:
                skip -= 1
                continue
            vulnerable.append(entry["func"].replace("\n\n", "\n"))
            keep -= 1
            if keep <= 0:
                break
    return vulnerable
