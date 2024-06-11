import json
from typing import List


def load_code_x_glue(skip: int = 0, keep: int = -1):
    file = open("../../../function.json")
    content = file.read()
    dataset = json.loads(content)
    vulnerable: List[str] = []
    if keep < 0:
        keep = len(dataset)
    i = 0
    for entry in dataset:
        if entry["target"] == 1:
            if skip > 0:
                skip -= 1
                continue
            vulnerable.append(entry["func"].replace("\n\n", "\n"))
            i += 1
            if i >= keep:
                break
    return vulnerable
