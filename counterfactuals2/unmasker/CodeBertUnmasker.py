from counterfactuals2.unmasker.AbstractUnmasker import AbstractUnmasker

from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline


class CodeBertUnmasker(AbstractUnmasker):
    model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    def get_mask(self) -> str:
        return "<mask>"

    def get_mask_replacement(self, code: str) -> str:
        result = self.fill_mask(code)
        while isinstance(result, list):
            result = result[0]
        return result["token_str"]
