class EvaluationResult:
    def __init__(self, json_entry):
        self.input_id = str(json_entry["input_id"])
        self.input_token_length = int(json_entry["input_token_length"])
        self.parameters = json_entry["parameters"]
        self.search_duration = float(json_entry["search_duration"])
        self.counterfactuals = json_entry["counterfactuals"]
        self.search_algorithm = json_entry["search_algorithm"]
        self.classifier = json_entry["classifier"]
        self.truncated = bool(json_entry["truncated"])
        self.perturber = json_entry["perturber"]
        self.tokenizer = json_entry["tokenizer"]
        self.unmasker = json_entry["unmasker"]
        if "cause" in json_entry:
            self.cause = json_entry["cause"]
        else:
            self.cause = None
        if "classification" in json_entry:
            self.classification = json_entry["classification"]
        else:
            self.classification = None

    def is_exception(self):
        return self.cause is not None

    def is_invalid_classification(self):
        return self.classification is not None

    def has_counterfactuals(self):
        return len(self.counterfactuals) > 0

    def get_input(self, inputs):
        return inputs[self.input_id]


class EvaluationParameters:
    def __init__(self, result: EvaluationResult):
        self.parameters = result.parameters
        self.search_algorithm = result.search_algorithm
        self.classifier = result.classifier
        self.perturber = result.perturber
        self.tokenizer = result.tokenizer
        self.unmasker = result.unmasker
        params = ""
        for (key, value) in result.parameters.items():
            params += str(key) + ":" + str(value) + ", "
        params = params.strip(", ")
        self.params_str = params
        self.hash = hash((params,
                          self.search_algorithm,
                          self.classifier,
                          self.perturber,
                          self.tokenizer,
                          self.unmasker))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        other_result: EvaluationParameters = other
        return self.parameters == other_result.parameters and \
               self.search_algorithm == other_result.search_algorithm and \
               self.classifier == other_result.classifier and \
               self.perturber == other_result.perturber and \
               self.tokenizer == other_result.tokenizer

    def __hash__(self):
        return self.hash

    def __str__(self):
        return str(self.__dict__)

    def get_human_readable_name(self):
        return (self.search_algorithm + " " +
                self.unmasker + " " +
                self.perturber + " " +
                self.classifier + " " +
                self.tokenizer + " " +
                self.params_str.replace(":", " ")).replace("NotApplicable ", "")


class EvaluationData:
    def __init__(self, evaluation_result: EvaluationResult):
        self.input_id = evaluation_result.input_id
        self.search_duration = evaluation_result.search_duration
        self.counterfactuals = evaluation_result.counterfactuals
        self.truncated = evaluation_result.truncated
        self.cause = evaluation_result.cause
        self.classification = evaluation_result.classification
        self.input_token_length = evaluation_result.input_token_length

    def is_exception(self):
        return self.cause is not None

    def is_invalid_classification(self):
        return self.classification is not None

    def has_counterfactuals(self):
        return len(self.counterfactuals) > 0

    def get_input(self, inputs):
        return inputs[int(self.input_id)]
