from typing import List

from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline

from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker


class CodeBertUnmasker(AbstractUnmasker):
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

    def __init__(self, device):
        super().__init__()
        self.model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(device)
        self.fill_mask = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer,device=int(str(device).replace("cuda:","")))
        self.device = device

    def get_mask(self) -> str:
        return "<mask>"

    def get_mask_replacement(self, original_token_id: int, code: str, dictionary: List[str]) -> int:

        tokens = self.tokenizer(code, return_tensors="pt")["input_ids"][0]
        o_len=len(tokens)

        if len(tokens) > 512:
            max = 0
            while True:
                tokens =  self.tokenizer(code, return_tensors="pt", truncation=True)["input_ids"][0]

                if max > 0:
                    tokens = list(tokens)
                    for i in range(max+1):
                        del tokens[-2]

                code = self.tokenizer.decode(tokens)
                tokens = self.tokenizer(code, return_tensors="pt")["input_ids"][0]

                if len(tokens) <=512:
                    break

                max+=1
                if max>5:
                    raise Exception()

        try:
            result = self.fill_mask(code)
        except Exception as e:
            if str(e).startswith("No mask_token"):
                raise e
            print(e)
            print("error in unmasking on code",code)
            print("original token length",o_len)
            print("length of untruncated",len(self.tokenizer(code, return_tensors="pt")["input_ids"][0]))
            print("actual tokens",len(tokens))
            raise e

        while isinstance(result, list):
            result = result[0]
        new_token = result["token_str"]

        if new_token not in dictionary:
            old_len = len(dictionary)
            dictionary.append(new_token)
            return old_len
        else:
            return dictionary.index(new_token)
