import time
from typing import List

from common.code_formatter import format_code
from counterfactuals2.classifier.AbstractClassifier import AbstractClassifier
from counterfactuals2.misc.Counterfactual import Counterfactual
import torch
from captum.attr import LayerIntegratedGradients
from captum.attr import TokenReferenceBase

from counterfactuals2.misc.LigSearchParameters import LigSearchParameters
from counterfactuals2.misc.SearchParameters import SearchParameters
from counterfactuals2.misc.SearchResults import SearchResult, SearchError, InvalidClassificationResult, NotApplicable
from counterfactuals2.misc.language import Language
from counterfactuals2.searchAlgorithms.AbstractSearchAlgorithm import AbstractSearchAlgorithm


class LigSearch(AbstractSearchAlgorithm):
    def __init__(
            self,
            classifier: AbstractClassifier,
            max_iterations: int = 100,
            steps_per_iteration: int = 20,
            recompute_attributions_for_each_iteration: bool = True,
            verbose: bool = False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            max_tokens_removal_ratio: float = .6
    ):
        super().__init__(None, classifier, verbose)
        self.max_tokens_removal_ratio = max_tokens_removal_ratio
        self.max_iterations = max_iterations
        self.steps_per_iteration = steps_per_iteration
        self.recompute_attributions_for_each_iteration = recompute_attributions_for_each_iteration
        self.lig = LayerIntegratedGradients(self.predict, self.classifier.get_embeddings())
        self.token_reference = TokenReferenceBase(reference_token_idx=self.classifier.get_padding_token_id())
        self.device = device
        self.classifier.prepare_for_lig(device)
        self.verbose = verbose

    def get_parameters(self) -> SearchParameters:
        return LigSearchParameters(self.max_iterations, self.steps_per_iteration, self.recompute_attributions_for_each_iteration, self.max_tokens_removal_ratio)

    def search(self, source_code: str) -> SearchResult:
        """Performs the search for counterfactuals. Do not override this function"""
        start = time.time()
        truncated = False
        na = NotApplicable()
        number_of_tokens_in_src = 0
        try:
            torch.cuda.empty_cache()
            tokens = self.classifier.tokenize(source_code)
            number_of_tokens_in_src = len(tokens)

            max_tokens = self.classifier.get_max_tokens()
            if number_of_tokens_in_src > max_tokens:
                if self.verbose:
                    print("input too long, truncating from " + str(number_of_tokens_in_src) + " tokens to " + str(max_tokens))
                truncated = True
                tokens = tokens[0:max_tokens - 1]
                source_code = ""
                for t in tokens:
                    source_code += self.classifier.token_id_to_string(t) + " "
                source_code = format_code(source_code, Language.Cpp)

            original_class, original_confidence = self.classifier.classify(source_code)

            if self.verbose:
                print("input classified as", original_class, "with a confidence of", original_confidence)

            if original_class:
                return InvalidClassificationResult(source_code, number_of_tokens_in_src, original_class, self, self.classifier, na, na, na, 0, self.get_parameters(), truncated)

            result = self.perform_lig_search(source_code, original_class)

            end = time.time()
            if self.verbose:
                print("search took", end - start, "seconds")
            return SearchResult(source_code, number_of_tokens_in_src, result, self, self.classifier, na, na, na, end - start, self.get_parameters(), truncated)
        except Exception as e:
            if self.verbose:
                print(e)
            end = time.time()
            return SearchError(source_code, number_of_tokens_in_src, e, self, self.classifier, na, na, na, end - start, self.get_parameters(), truncated)

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

    def get_input_ids(self, source_code: str):
        tokenized = self.classifier.tokenize(source_code)
        input_ids = tokenized["input_ids"]
        bos_id = self.classifier.get_begin_of_string_token_id()
        eos_id = self.classifier.get_end_of_string_token_id()
        if input_ids[0] != bos_id:
            input_ids.insert(0, bos_id)

        if len(input_ids) <= 2:
            return [], []

        i = 0
        eos_index = len(input_ids) - 1
        for token in input_ids:
            if token == eos_id:
                eos_index = i
                break
            i += 1

        while len(input_ids) > self.classifier.get_max_tokens():
            del input_ids[eos_index - 1]
            eos_index -= 1
        return input_ids, eos_index

    def get_lig_attributes(self, source_code: str, target_class) -> (List[int], List):
        input_ids, eos_index = self.get_input_ids(source_code)

        input_id_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(0)
        reference_indices = self.get_reference_indices(len(input_ids), eos_index)

        lig_attributions, delta = self.lig_attribute(input_id_tensor.to(self.device), reference_indices.unsqueeze(0).to(self.device), target_class)
        attributions = lig_attributions.squeeze(0).tolist()

        del input_ids[0]
        del attributions[0]  # remove begin of string token

        if input_ids[-1] == self.classifier.get_end_of_string_token_id():
            del input_ids[-1]
            del attributions[-1]  # remove end of string token

        return input_ids, attributions

    def eliminate_highest_attribution(self, input_ids, attributes) -> int:
        best_index = -1
        best_attribution = -999999

        for i in range(len(input_ids)):
            if attributes[i] > best_attribution:
                best_index = i
                best_attribution = attributes[i]

        if self.verbose:
            print("deleting token ", input_ids[best_index], ":", self.classifier.token_id_to_string(input_ids[best_index]), "with attribution", attributes[best_index])

        token_id = input_ids[best_index]

        del input_ids[best_index]
        del attributes[best_index]

        return token_id

    def perform_lig_search(self, source_code, original_class) -> List[Counterfactual]:
        counterfactuals: List[Counterfactual] = []
        target_class = not original_class
        iterations = 0
        start_time = time.time()
        number_of_tokens_in_input = -1
        number_of_changes = 0

        changed_values: List[int] = []

        if not self.recompute_attributions_for_each_iteration:
            input_ids, attributes = self.get_lig_attributes(source_code, target_class)
            original_number_of_tokens = number_of_tokens_in_input = len(input_ids)
            if len(input_ids) < 1:
                return counterfactuals
        else:
            input_ids, _ = self.get_input_ids(source_code)
            original_number_of_tokens = len(input_ids)
        abort_when_less_than_tokens = self.max_tokens_removal_ratio * original_number_of_tokens

        while iterations < self.max_iterations:
            torch.cuda.empty_cache()
            if self.verbose:
                print("iteration #", iterations, end="")
            iterations += 1

            if self.recompute_attributions_for_each_iteration:
                input_ids, attributes = self.get_lig_attributes(source_code, target_class)
                if number_of_tokens_in_input < 0:
                    number_of_tokens_in_input = len(input_ids)

            if len(input_ids) <= 1:  # one element will be removed in the next line
                return counterfactuals

            changed_values.append(self.eliminate_highest_attribution(input_ids, attributes))
            number_of_changes += 1

            if self.verbose:
                print(" with", len(input_ids), "tokens remaining")

            src = ""
            for id in input_ids:
                str = self.classifier.token_id_to_string(id)
                for c in str:
                    if c.isascii():
                        src += c
                    elif c == '\u2581' or c == '\u0120' or c == '\u010a':
                        src += ' '
                    elif self.verbose:
                        print("Unknown character", c.encode("unicode_escape"))

            source_code = format_code(src, Language.Cpp)
            if self.verbose:
                print(source_code)

            new_class, new_confidence = self.classifier.classify(source_code)

            if new_class != original_class:
                changed_lines = [self.classifier.token_id_to_string(i) for i in changed_values]
                counterfactuals.append(Counterfactual(source_code, iterations, start_time, number_of_tokens_in_input, number_of_changes, len(input_ids), changed_lines))
                return counterfactuals
            if len(input_ids) < abort_when_less_than_tokens:
                if self.verbose:
                    print("too little tokens remaining, aborting search")
                return counterfactuals
        return counterfactuals
