import json
from typing import List


def load_code_x_glue(keep: int = -1):
    file = open("D:/A_Uni/A_MasterThesis/CodeXGlue/function.json")
    content = file.read()
    dataset = json.loads(content)
    vulnerable: List[str] = []
    if keep < 0:
        keep = len(dataset)
    i = 0
    for entry in dataset:
        if entry["target"] == 1:
            vulnerable.append(entry["func"].replace("\n\n", "\n"))
            i += 1
            if i > keep:
                break
    return vulnerable
