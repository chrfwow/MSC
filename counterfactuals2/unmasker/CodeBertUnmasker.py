from typing import List

from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline

from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker


class CodeBertUnmasker(AbstractUnmasker):
    model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    def get_mask(self) -> str:
        return "<mask>"

    def get_mask_replacement(self, original_token_id: int, code: str, dictionary: List[str]) -> int:
        result = self.fill_mask(code)
        while isinstance(result, list):
            result = result[0]
        new_token = result["token_str"]
        old_len = len(dictionary)
        dictionary.append(new_token)
        return old_len
