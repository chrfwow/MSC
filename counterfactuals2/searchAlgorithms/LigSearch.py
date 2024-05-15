import time
from typing import List

from common.code_formatter import format_code
from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from counterfactuals2.misc.Counterfactual import Counterfactual
import torch
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization, TokenReferenceBase

from counterfactuals2.misc.SearchResults import SearchResult
from counterfactuals2.misc.language import Language
from counterfactuals2.searchAlgorithms.AbstractSearchAlgorithm import AbstractSearchAlgorithm


class LigSearch(AbstractSearchAlgorithm):
    def __init__(self, classifier: AbstractClassifier, max_iterations: int = 100, steps_per_iteration: int = 50, recompute_attributions_for_each_iteration: bool = True):
        super().__init__(None, classifier)
        self.max_iterations = max_iterations
        self.steps_per_iteration = steps_per_iteration
        self.recompute_attributions_for_each_iteration = recompute_attributions_for_each_iteration
        self.lig = LayerIntegratedGradients(self.predict, self.classifier.get_embeddings())
        self.token_reference = TokenReferenceBase(reference_token_idx=self.classifier.get_padding_token_id())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.prepare_for_lig(self.device)

    def search(self, source_code: str) -> SearchResult:
        """Performs the search for counterfactuals. Do not override this function"""
        start = time.time()
        original_class, original_confidence = self.classifier.classify(source_code)

        print("input classified as", original_class, "with a confidence of", original_confidence)

        result = self.perform_lig_search(source_code, original_class)

        end = time.time()
        print("search took", end - start, "seconds")
        return SearchResult(source_code, result, self, self.classifier, None, None, None, end - start)

    def predict(self, inputs):
        attention_mask = torch.where(inputs == self.classifier.get_padding_token_id(), 0, 1)
        return self.classifier.get_logits(input_indices=inputs, attention_mask=attention_mask)

    def get_reference_indices(self, input_length, eos_index):
        reference_indices = self.token_reference.generate_reference(input_length, device=self.device)
        reference_indices[0] = self.classifier.get_begin_of_string_token_id()
        reference_indices[eos_index] = self.classifier.get_end_of_string_token_id()
        return reference_indices

    def lig_attribute(self, input_indices, baseline, target):
        if target:
            t = 0
        else:
            t = 1
        attributions, delta = self.lig.attribute(input_indices, baselines=baseline, target=t, n_steps=self.steps_per_iteration, return_convergence_delta=True)
        attributions = attributions.sum(dim=-1)
        return attributions / torch.norm(attributions), delta

    def get_lig_attributes(self, source_code: str, target_class) -> (List[int], List):
        tokenized = self.classifier.tokenize(source_code)
        input_ids = tokenized["input_ids"]
        bos_id = self.classifier.get_begin_of_string_token_id()
        eos_id = self.classifier.get_end_of_string_token_id()
        if input_ids[0] != bos_id:
            input_ids.insert(0, bos_id)
        input_id_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(0)

        if len(input_ids) <= 2:
            return [], []

        token_strings = []
        i = 0
        eos_index = len(input_ids) - 1
        for token in input_ids:
            if token == eos_id:
                eos_index = i
            i += 1

            token_strings.append(self.classifier.token_id_to_string(token))

        reference_indices = self.get_reference_indices(len(input_ids), eos_index)

        lig_attributions, delta = self.lig_attribute(input_id_tensor, reference_indices.unsqueeze(0), target_class)
        attributions = lig_attributions.squeeze(0).tolist()

        del input_ids[0]
        del attributions[0]  # remove begin of string token

        if input_ids[-1] == eos_id:
            del input_ids[-1]
            del attributions[-1]  # remove end of string token

        return input_ids, attributions

    def eliminate_highest_attribution(self, input_ids, attributes):
        best_index = -1
        best_attribution = -999999

        for i in range(len(input_ids)):
            if attributes[i] > best_attribution:
                best_index = i
                best_attribution = attributes[i]

        print("deleting token ", input_ids[best_index], ":", self.classifier.token_id_to_string(input_ids[best_index]), "with attribution", attributes[best_index])

        del input_ids[best_index]
        del attributes[best_index]

    def perform_lig_search(self, source_code, original_class) -> List[Counterfactual]:
        counterfactuals: List[Counterfactual] = []
        target_class = not original_class
        iterations = 0

        if not self.recompute_attributions_for_each_iteration:
            input_ids, attributes = self.get_lig_attributes(source_code, target_class)
            if len(input_ids) < 1:
                return counterfactuals

        while iterations < self.max_iterations:
            print("iteration #", iterations, end="")
            iterations += 1

            if self.recompute_attributions_for_each_iteration:
                input_ids, attributes = self.get_lig_attributes(source_code, target_class)

            if len(input_ids) <= 1:  # one element will be removed in the next line
                return counterfactuals

            self.eliminate_highest_attribution(input_ids, attributes)

            print(" with", len(input_ids), "tokens remaining")

            src = ""
            for id in input_ids:
                str = self.classifier.token_id_to_string(id)
                for c in str:
                    if c.isascii():
                        src += c
                    elif c == '\u2581' or c == '\u0120' or c == '\u010a':
                        src += ' '
                    else:
                        print("Unknown character", c.encode("unicode_escape"))

            source_code = format_code(src, Language.Cpp)
            print(source_code)

            new_class, new_confidence = self.classifier.classify(source_code)

            if new_class != original_class:
                counterfactuals.append(Counterfactual(source_code, iterations))
                return counterfactuals

        return counterfactuals
