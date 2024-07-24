import json
import matplotlib
from matplotlib.ticker import MaxNLocator
import jellyfish
import re

from counterfactuals2.evaluation.id_finder import find
from counterfactuals2.misc.DatasetLoader import load_code_x_glue

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from counterfactuals2.evaluation.EvaluationResult import EvaluationResult, EvaluationParameters, EvaluationData

results_path = "results/"


def eval(path):
    if type(path) != list:
        path = [path]

    first_doc = path[0]

    highest_input_id = -1
    total_number_of_evaluations_per_classifier = dict()
    total_number_of_invalid_evaluations_per_classifier = dict()

    with open(first_doc) as file:
        content = file.read()
    data = json.loads(content)
    duration = data["duration_sec"]
    inputs = data["ids_of_input"]
    raw_results = data["results"]
    total_number_of_evaluations_per_classifier[raw_results[0]["classifier"]] = 1
    total_number_of_invalid_evaluations_per_classifier[raw_results[0]["classifier"]] = 1
    input_id = find(raw_results, len(path))
    for r in raw_results:
        r["file_id"] = 0
        r["input_id"] = input_id
        r["file_name"] = first_doc
    if input_id > highest_input_id:
        highest_input_id = input_id

    index = 0
    for p in path[1:]:
        index += 1
        with open(p) as f:
            content = f.read()
            d = json.loads(content)
            duration += d["duration_sec"]
            for i, inp in d["ids_of_input"].items():
                inputs[i] = inp
            res = d["results"]
            input_id = find(res, len(path))
            if input_id > highest_input_id:
                highest_input_id = input_id
            added_classifiers = set()
            classifiers_with_invalids = set()
            for r in res:
                r["file_id"] = index
                r["input_id"] = input_id
                r["file_name"] = p
                raw_results.append(r)
                classifier = r["classifier"]
                if classifier not in added_classifiers:
                    added_classifiers.add(classifier)
                    if classifier in total_number_of_evaluations_per_classifier:
                        total_number_of_evaluations_per_classifier[classifier] = total_number_of_evaluations_per_classifier[classifier] + 1
                    else:
                        total_number_of_evaluations_per_classifier[classifier] = 1
                if "classification" in r:
                    if classifier not in classifiers_with_invalids:
                        classifiers_with_invalids.add(classifier)
                        if classifier in total_number_of_invalid_evaluations_per_classifier:
                            total_number_of_invalid_evaluations_per_classifier[classifier] = total_number_of_invalid_evaluations_per_classifier[classifier] + 1
                        else:
                            total_number_of_invalid_evaluations_per_classifier[classifier] = 1

    # vulberta had a bug which was fixed in the end of june, end the next available slots on the server were in july
    raw_results = list(filter(lambda x: x["classifier"] != "VulBERTa_MLP_Classifier" or "_2024_June_" not in x["file_name"], raw_results))

    raw_inputs = load_code_x_glue(keep=highest_input_id + 1)
    for i, s in enumerate(raw_inputs):
        inputs[i] = s

    plot_cf_sizes_per_algo(raw_results)

    print("duration", duration)
    results = dict()
    for raw in raw_results:
        eval_result = EvaluationResult(raw)
        params = EvaluationParameters(eval_result)
        data = EvaluationData(eval_result)
        results.setdefault(params, []).append(data)

    similarities = get_average_similarity(results, inputs)
    plot_similarities(similarities)

    plot_times_per_search_algo(raw_results)
    plot_lig(raw_results)

    names = get_and_print_names(results)

    times = time_per_token(results)
    numbers_of_cf = number_of_cf(results)
    absolute_small_changes = get_cf_with_absolute_little_changes(results, names)
    relative_small_changes = get_cf_with_relative_little_changes(results, .2)
    removed = get_removed_types_on_small_change_cf(absolute_small_changes)

    # print(times)
    # print(numbers_of_cf)
    # print(absolute_small_changes)
    # print(relative_small_changes)
    # print(removed)
    # print(similarities)

    print("pre plot")
    plot_histogram_how_many_cf_with_how_many_changes(results, names)
    print()
    plot_times(times, names)

    print("calculating similarities, this could take a while...")

    print("\n")
    print_table(results, names, times, numbers_of_cf, absolute_small_changes, relative_small_changes, similarities)
    print("\n")
    print_parameters(results, names)
    print("\n")
    print_removed_types(results, removed, names)
    print("\n")
    print_invalid_classifications(total_number_of_evaluations_per_classifier, total_number_of_invalid_evaluations_per_classifier)


def plot_cf_sizes_per_algo(raw_results):
    x_axis = ["LineTokenizer", "ClangTokenizer", "MaskedPerturber", "MutationPerturber", "RemoveTokenPerturber"]
    classifiers = ["CodeBertClassifier", "PLBartClassifier", "CodeT5Classifier", "VulBERTa_MLP_Classifier"]

    search_algos = dict()
    lig = dict()

    true = dict()
    true["total_cf_entries"] = 0
    true["total_cfs"] = 0
    true["total_changed_tokens"] = 0
    true["total_input_tokens"] = 0
    false = dict()
    false["total_cf_entries"] = 0
    false["total_cfs"] = 0
    false["total_changed_tokens"] = 0
    false["total_input_tokens"] = 0

    lig["True"] = true
    lig["False"] = false

    for result in raw_results:
        if "cause" in result or "classification" in result:
            continue
        counterfactuals = result["counterfactuals"]
        number_of_cfs = len(counterfactuals)

        search_algorithm = result["search_algorithm"]
        if search_algorithm == "LigSearch":
            if number_of_cfs > 0:

                if result["parameters"]["recompute_attributions_for_each_iteration"]:
                    data = lig["True"]
                else:
                    data = lig["False"]

                data["total_cfs"] = data["total_cfs"] + number_of_cfs
                data["total_cf_entries"] = data["total_cf_entries"] + 1

                for c in counterfactuals:
                    data["total_changed_tokens"] = data["total_changed_tokens"] + c['number_of_changes']
                    data["total_input_tokens"] = data["total_input_tokens"] + c['number_of_tokens_in_input']

            continue
        if search_algorithm not in search_algos:
            data = dict()
            for x in x_axis:
                t = dict()
                t["total_cf_entries"] = 0
                t["total_cfs"] = 0
                t["total_changed_tokens"] = 0
                t["total_input_tokens"] = 0
                data[x] = t
            for c in classifiers:
                t = dict()
                t["total_cf_entries"] = 0
                t["total_cfs"] = 0
                t["total_changed_tokens"] = 0
                t["total_input_tokens"] = 0
                data[c] = t
            search_algos[search_algorithm] = data
        else:
            data = search_algos[search_algorithm]

        tokenizer = result["tokenizer"]
        perturber = result["perturber"]
        classifier = result["classifier"]

        data[tokenizer]["total_cf_entries"] = data[tokenizer]["total_cf_entries"] + 1
        data[tokenizer]["total_cfs"] = data[tokenizer]["total_cfs"] + number_of_cfs

        data[perturber]["total_cf_entries"] = data[perturber]["total_cf_entries"] + 1
        data[perturber]["total_cfs"] = data[perturber]["total_cfs"] + number_of_cfs

        data[classifier]["total_cf_entries"] = data[classifier]["total_cf_entries"] + 1
        data[classifier]["total_cfs"] = data[classifier]["total_cfs"] + number_of_cfs

        for c in counterfactuals:
            data[tokenizer]["total_changed_tokens"] = data[tokenizer]["total_changed_tokens"] + c['number_of_changes']
            data[tokenizer]["total_input_tokens"] = data[tokenizer]["total_input_tokens"] + c['number_of_tokens_in_input']
            data[perturber]["total_changed_tokens"] = data[perturber]["total_changed_tokens"] + c['number_of_changes']
            data[perturber]["total_input_tokens"] = data[perturber]["total_input_tokens"] + c['number_of_tokens_in_input']
            data[classifier]["total_changed_tokens"] = data[classifier]["total_changed_tokens"] + c['number_of_changes']
            data[classifier]["total_input_tokens"] = data[classifier]["total_input_tokens"] + c['number_of_tokens_in_input']

    import numpy as np
    cm = 1 / 2.54  # centimeters in inches

    # relative number of changes per search algo

    this_x_axis = list(x_axis)
    x = np.arange(len(this_x_axis) + 2)
    width = .25
    multiplier = 0
    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    this_search_algos = dict(search_algos)
    this_search_algos["LigSearch"] = lig

    bars = []

    for s, data in search_algos.items():
        d = dict()
        d[s] = []
        for a in this_x_axis:
            d[s].append(round(data[a]["total_changed_tokens"] / data[a]["total_input_tokens"] * 100, 1))

        d[s].append(0)
        d[s].append(0)

        offset = width * multiplier
        rects = plt.bar(
            x + offset,
            d[s],
            width,
            label=s.replace("_", " ").replace("Classifier", "").strip()
        )
        bars.append(ax.bar_label(rects, padding=3))
        multiplier += 1

    items = [true, false]
    for t in items:
        if t == true:
            label = "Recompute"
            ind = 5
        else:
            label = "Don't Recompute"
            ind = 6

        vals = [0] * 7
        vals[ind] = round(t["total_changed_tokens"] / t["total_input_tokens"] * 100, 1)

        offset = width * multiplier
        rects = plt.bar(
            x + width,
            vals,
            width,
            label=label
        )
        bars.append(ax.bar_label(rects, padding=3))
        multiplier += 1

    ax.set_yscale("log")
    # ax.legend([*[s for s,d in search_algos.items()], "Lig Search Recompute", "Lig Search Don't Recompute"])

    # Greedy
    m1, = ax.plot([], [], c='steelblue', marker='s', markersize=10, fillstyle='left', linestyle='none')
    m2, = ax.plot([], [], c='steelblue', marker='s', markersize=10, fillstyle='right', linestyle='none')

    # genetic
    m3, = ax.plot([], [], c='darkorange', marker='s', markersize=10, fillstyle='left', linestyle='none')
    m4, = ax.plot([], [], c='darkorange', marker='s', markersize=10, fillstyle='right', linestyle='none')

    # kees
    m5, = ax.plot([], [], c='forestgreen', marker='s', markersize=10, fillstyle='left', linestyle='none')
    m6, = ax.plot([], [], c='forestgreen', marker='s', markersize=10, fillstyle='right', linestyle='none')

    # lig
    m7, = ax.plot([], [], c='red', marker='s', markersize=10, fillstyle='left', linestyle='none')
    m8, = ax.plot([], [], c='purple', marker='s', markersize=10, fillstyle='right', linestyle='none')

    # ---- Plot Legend ----
    ax.legend(((m1, m2), (m3, m4), (m5, m6), (m7, m8)), (*[s for s, d in search_algos.items()], "Lig Search"), numpoints=1, loc="lower right")

    ax.set_ylabel("Average percentage of changed tokens [%]")
    ax.set_title("Average percentage of input tokens changed per counterfactual")
    ax.set_xticks(x + width, [x.replace("Perturber", "").replace("Tokenizer", "") for x in [*x_axis, "Recompute", "Don't Recompute"]])
    # ax.set_xticks(x + width, x_axis)

    plt.savefig(results_path + "avg_size_of_cfs_per_search_algo.png")

    # plt.show()

    plt.close(fig)


def plot_similarities(similarities):
    x_axis = ["LineTokenizer", "ClangTokenizer", "MaskedPerturber", "MutationPerturber", "RemoveTokenPerturber", "Recompute", "Don't Recompute"]
    classifiers = ["CodeBertClassifier", "PLBartClassifier", "CodeT5Classifier", "VulBERTa_MLP_Classifier"]
    search_algos = dict()

    true = dict()
    true["sum"] = 0.0
    true["entries"] = 0
    false = dict()
    false["sum"] = 0.0
    false["entries"] = 0

    for sim, val in similarities.items():
        search_algorithm = sim.search_algorithm
        classifier = sim.classifier
        perturber = sim.perturber
        tokenizer = sim.tokenizer

        if search_algorithm not in search_algos:
            data = dict()
            for x in x_axis:
                t = dict()
                t["sim_sum"] = 0.0
                t["sim_entries"] = 0
                data[x] = t
            for x in classifiers:
                t = dict()
                t["sim_sum"] = 0.0
                t["sim_entries"] = 0
                data[x] = t
            data["total_entries"] = 0
            search_algos[search_algorithm] = data
        else:
            data = search_algos[search_algorithm]

        data["total_entries"] = data["total_entries"] + 1

        if val == 0.0:
            continue

        if search_algorithm == "LigSearch":
            if sim.parameters["recompute_attributions_for_each_iteration"]:
                true["sum"] = true["sum"] + val
                true["entries"] = true["entries"] + 1
            else:
                false["sum"] = false["sum"] + val
                false["entries"] = false["entries"] + 1

        data[classifier]["sim_sum"] = data[classifier]["sim_sum"] + val
        data[classifier]["sim_entries"] = data[classifier]["sim_entries"] + 1

        if perturber != "NotApplicable":
            data[perturber]["sim_sum"] = data[perturber]["sim_sum"] + val
            data[perturber]["sim_entries"] = data[perturber]["sim_entries"] + 1

        if tokenizer != "NotApplicable":
            data[tokenizer]["sim_sum"] = data[tokenizer]["sim_sum"] + val
            data[tokenizer]["sim_entries"] = data[tokenizer]["sim_entries"] + 1

    import numpy as np
    cm = 1 / 2.54  # centimeters in inches

    x = np.arange(len(x_axis))
    width = .25
    multiplier = 0
    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    bars = []

    for s, data in search_algos.items():
        d = dict()
        d[s] = []
        for a in x_axis[:-2]:
            if data[a]["sim_entries"] == 0:
                d[s].append(-1)
            else:
                d[s].append(round(data[a]["sim_sum"] / data[a]["sim_entries"], 2))

        d[s].append(-1)
        d[s].append(-1)

        offset = width * multiplier
        rects = plt.bar(
            x + offset,
            d[s],
            width,
            label=s.replace("_", " ").replace("Classifier", "").replace("Perturber", "").replace("Tokenizer", "").strip()
        )
        bars.append(ax.bar_label(rects, padding=3))
        multiplier += 1

    items = [true, false]
    for t in items:
        if t == true:
            label = "Recompute"
            ind = 5
        else:
            label = "Don't Recompute"
            ind = 6

        vals = [-1] * 7
        if t["entries"] == 0:
            vals[ind] = -1
        else:
            vals[ind] = round(t["sum"] / t["entries"], 2)

        offset = width * multiplier
        rects = plt.bar(
            x + width,
            vals,
            width,
            label=label
        )
        bars.append(ax.bar_label(rects, padding=3))
        multiplier += 1

    # ax.set_yscale("log")
    ax.set_ylim(0)

    m1, = ax.plot([], [], c='steelblue', marker='s', markersize=10, fillstyle='left', linestyle='none')
    m2, = ax.plot([], [], c='steelblue', marker='s', markersize=10, fillstyle='right', linestyle='none')

    # genetic
    m3, = ax.plot([], [], c='darkorange', marker='s', markersize=10, fillstyle='left', linestyle='none')
    m4, = ax.plot([], [], c='darkorange', marker='s', markersize=10, fillstyle='right', linestyle='none')

    # kees
    m5, = ax.plot([], [], c='forestgreen', marker='s', markersize=10, fillstyle='left', linestyle='none')
    m6, = ax.plot([], [], c='forestgreen', marker='s', markersize=10, fillstyle='right', linestyle='none')

    # lig
    m7, = ax.plot([], [], c='red', marker='s', markersize=10, fillstyle='left', linestyle='none')
    m8, = ax.plot([], [], c='purple', marker='s', markersize=10, fillstyle='right', linestyle='none')

    # ---- Plot Legend ----
    ax.legend(((m1, m2), (m3, m4), (m5, m6), (m7, m8)), (*[s for s, d in search_algos.items()], "Lig Search"), numpoints=1)

    ax.set_ylabel("Average string similarity")
    ax.set_title("Average string similarity for all Classifiers")
    ax.set_xticks(x + width, [x.replace("Perturber", "").replace("Tokenizer", "") for x in x_axis])
    # ax.set_xticks(x + width, x_axis)
    # plt.show()
    plt.savefig(results_path + "sim.png")

    plt.close(fig)


def plot_lig(raw_results):
    classifiers = ["CodeBertClassifier", "PLBartClassifier", "VulBERTa_MLP_Classifier"]
    lig = dict()
    for result in raw_results:
        search_algorithm = result["search_algorithm"]
        if search_algorithm == "LigSearch":
            use = lig
        else:
            continue

        classifier = result["classifier"]
        if classifier not in use:
            t = dict()
            t["total_cf_entries"] = 0
            t["total_cfs"] = 0
            t["duration_per_cf"] = 0.0
            t["total_entries"] = 0
            use[classifier] = t

        use[classifier]["total_entries"] = use[classifier]["total_entries"] + 1
        cfs = result["counterfactuals"]
        num_cfs = len(cfs)
        if num_cfs == 0:
            continue
        use[classifier]["total_cf_entries"] = use[classifier]["total_cf_entries"] + 1
        use[classifier]["total_cfs"] = use[classifier]["total_cfs"] + num_cfs
        use[classifier]["duration_per_cf"] = use[classifier]["duration_per_cf"] + float(result["search_duration"])

    import numpy as np
    x = np.arange(len(classifiers))

    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    plt.bar(classifiers, [0 if lig[a]["total_cf_entries"] == 0 else round(lig[a]["duration_per_cf"] / lig[a]["total_cf_entries"], 2) for a in classifiers])

    # ax.set_yscale("log")
    ax.legend()
    ax.set_ylabel("Average search duration per counterfactual [s]")
    ax.set_title("Average search duration per counterfactual for LigSearch")
    ax.set_xticks(x, classifiers)
    # plt.show()
    plt.savefig(results_path + "time_per_cf_lig.png")

    plt.close(fig)

    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    plt.bar(classifiers, [0 if lig[a]["total_cf_entries"] == 0 else round(lig[a]["total_cfs"] / lig[a]["total_entries"], 2) for a in classifiers])

    # ax.set_yscale("log")
    ax.legend()
    ax.set_ylabel("Average number of counterfactuals")
    ax.set_title("Average number of counterfactuals per search run for each search algorithm")
    ax.set_xticks(x, classifiers)
    # plt.show()
    plt.savefig(results_path + "avg_num_of_cfs_lig.png")

    plt.close(fig)


def plot_times_per_search_algo(raw_results):
    x_axis = ["LineTokenizer", "ClangTokenizer", "MaskedPerturber", "MutationPerturber", "RemoveTokenPerturber"]
    classifiers = ["CodeBertClassifier", "PLBartClassifier", "CodeT5Classifier", "VulBERTa_MLP_Classifier"]

    search_algos = dict()
    lig = dict()
    lig["total_cfs"] = 0
    lig["total_cf_entries"] = 0
    lig["total_entries"] = 0
    lig["total_duration"] = 0.0

    true = dict()
    true["shortest_cf_duration"] = 0
    true["total_cf_entries"] = 0
    false = dict()
    false["shortest_cf_duration"] = 0
    false["total_cf_entries"] = 0

    lig["True"] = true
    lig["False"] = false

    for result in raw_results:
        if "cause" in result or "classification" in result:
            continue
        counterfactuals = result["counterfactuals"]
        number_of_cfs = len(counterfactuals)
        duration = float(result["search_duration"])

        search_algorithm = result["search_algorithm"]
        if search_algorithm == "LigSearch":
            if number_of_cfs > 0:
                lig["total_cfs"] = lig["total_cfs"] + number_of_cfs
                lig["total_cf_entries"] = lig["total_cf_entries"] + 1

                if result["parameters"]["recompute_attributions_for_each_iteration"]:
                    lig["True"]["shortest_cf_duration"] += counterfactuals[0]["duration"]
                    lig["True"]["total_cf_entries"] += 1
                else:
                    lig["False"]["shortest_cf_duration"] += counterfactuals[0]["duration"]
                    lig["False"]["total_cf_entries"] += 1

            lig["total_entries"] = lig["total_entries"] + 1
            lig["total_duration"] = lig["total_duration"] + duration

            continue
        if search_algorithm not in search_algos:
            data = dict()
            for x in x_axis:
                t = dict()
                t["total_input_token_entries"] = 0
                t["total_input_tokens"] = 0
                t["total_cf_entries"] = 0
                t["total_cfs"] = 0
                t["duration_per_token"] = 0.0
                t["duration_per_cf"] = 0.0
                t["shortest_cf_duration"] = 0
                data[x] = t
            for c in classifiers:
                t = dict()
                t["total_input_token_entries"] = 0
                t["total_input_tokens"] = 0
                t["total_cf_entries"] = 0
                t["total_cfs"] = 0
                t["duration_per_token"] = 0.0
                t["duration_per_cf"] = 0.0
                t["shortest_cf_duration"] = 0
                data[c] = t
            data["no_cfs"] = 0
            data["total_entries"] = 0
            data["total_cfs"] = 0
            data["total_duration"] = 0.0
            search_algos[search_algorithm] = data
        else:
            data = search_algos[search_algorithm]

        data["total_entries"] = data["total_entries"] + 1
        data["total_cfs"] = data["total_cfs"] + number_of_cfs
        data["total_duration"] = data["total_duration"] + duration

        tokenizer = result["tokenizer"]
        perturber = result["perturber"]
        classifier = result["classifier"]

        input_token_length = int(result["input_token_length"])

        if input_token_length == 0:
            continue

        duration_per_token = duration / input_token_length

        data[classifier]["total_input_token_entries"] = data[classifier]["total_input_token_entries"] + 1
        data[classifier]["total_input_tokens"] = data[classifier]["total_input_tokens"] + input_token_length
        data[classifier]["duration_per_token"] = data[classifier]["duration_per_token"] + duration_per_token

        if number_of_cfs == 0:
            data["no_cfs"] = data["no_cfs"] + 1
            # todo plot these occurrences as well
        else:
            duration_per_cf = duration / number_of_cfs
            data[classifier]["total_cf_entries"] = data[classifier]["total_cf_entries"] + 1
            data[classifier]["total_cfs"] = data[classifier]["total_cfs"] + number_of_cfs
            data[classifier]["duration_per_cf"] = data[classifier]["duration_per_cf"] + duration_per_cf

            current_shortest_duration = 9999999999
            for cf in counterfactuals:
                if cf["duration"] < current_shortest_duration:
                    current_shortest_duration = cf["duration"]
            if current_shortest_duration < 9999999999:
                data[classifier]["shortest_cf_duration"] += current_shortest_duration

        if tokenizer == "NotApplicable" or perturber == "NotApplicable":
            continue

        data[tokenizer]["total_input_token_entries"] = data[tokenizer]["total_input_token_entries"] + 1
        data[perturber]["total_input_token_entries"] = data[perturber]["total_input_token_entries"] + 1
        data[tokenizer]["total_input_tokens"] = data[tokenizer]["total_input_tokens"] + input_token_length
        data[perturber]["total_input_tokens"] = data[perturber]["total_input_tokens"] + input_token_length

        data[tokenizer]["duration_per_token"] = data[tokenizer]["duration_per_token"] + duration_per_token
        data[perturber]["duration_per_token"] = data[perturber]["duration_per_token"] + duration_per_token

        if number_of_cfs > 0:
            duration_per_cf = duration / number_of_cfs
            data[tokenizer]["duration_per_cf"] = data[tokenizer]["duration_per_cf"] + duration_per_cf
            data[perturber]["duration_per_cf"] = data[perturber]["duration_per_cf"] + duration_per_cf
            data[tokenizer]["total_cf_entries"] = data[tokenizer]["total_cf_entries"] + 1
            data[perturber]["total_cf_entries"] = data[perturber]["total_cf_entries"] + 1
            data[perturber]["total_cfs"] = data[perturber]["total_cfs"] + number_of_cfs
            data[tokenizer]["total_cfs"] = data[tokenizer]["total_cfs"] + number_of_cfs

            current_shortest_duration = 9999999999
            for cf in counterfactuals:
                if cf["duration"] < current_shortest_duration:
                    current_shortest_duration = cf["duration"]

            if current_shortest_duration < 9999999999:
                data[tokenizer]["shortest_cf_duration"] += current_shortest_duration
                data[perturber]["shortest_cf_duration"] += current_shortest_duration

    import numpy as np
    cm = 1 / 2.54  # centimeters in inches

    # Search duration of first cf for each config

    this_x_axis = list(x_axis)
    x = np.arange(len(this_x_axis) + 2)
    width = .25
    multiplier = 0
    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    this_search_algos = dict(search_algos)
    this_search_algos["LigSearch"] = lig

    bars = []

    for s, data in search_algos.items():
        d = dict()
        d[s] = []
        for a in this_x_axis:
            d[s].append(round(data[a]["shortest_cf_duration"] / data[a]["total_cf_entries"], 2))

        d[s].append(0)
        d[s].append(0)

        offset = width * multiplier
        rects = plt.bar(
            x + offset,
            d[s],
            width,
            label=s.replace("_", " ").replace("Classifier", "").strip()
        )
        bars.append(ax.bar_label(rects, padding=3))
        multiplier += 1

    items = [true, false]
    for t in items:
        if t == true:
            label = "Recompute"
            ind = 5
        else:
            label = "Don't Recompute"
            ind = 6

        vals = [0] * 7
        vals[ind] = round(t["shortest_cf_duration"] / t["total_cf_entries"], 2)

        offset = width * multiplier
        rects = plt.bar(
            x + width,
            vals,
            width,
            label=label
        )
        bars.append(ax.bar_label(rects, padding=3))
        multiplier += 1

    ax.set_yscale("log")
    # ax.legend([*[s for s,d in search_algos.items()], "Lig Search Recompute", "Lig Search Don't Recompute"])

    # Greedy
    m1, = ax.plot([], [], c='steelblue', marker='s', markersize=10, fillstyle='left', linestyle='none')
    m2, = ax.plot([], [], c='steelblue', marker='s', markersize=10, fillstyle='right', linestyle='none')

    # genetic
    m3, = ax.plot([], [], c='darkorange', marker='s', markersize=10, fillstyle='left', linestyle='none')
    m4, = ax.plot([], [], c='darkorange', marker='s', markersize=10, fillstyle='right', linestyle='none')

    # kees
    m5, = ax.plot([], [], c='forestgreen', marker='s', markersize=10, fillstyle='left', linestyle='none')
    m6, = ax.plot([], [], c='forestgreen', marker='s', markersize=10, fillstyle='right', linestyle='none')

    # lig
    m7, = ax.plot([], [], c='red', marker='s', markersize=10, fillstyle='left', linestyle='none')
    m8, = ax.plot([], [], c='purple', marker='s', markersize=10, fillstyle='right', linestyle='none')

    # ---- Plot Legend ----
    ax.legend(((m1, m2), (m3, m4), (m5, m6), (m7, m8)), (*[s for s, d in search_algos.items()], "Lig Search"), numpoints=1)

    ax.set_ylabel("Average search duration for the first counterfactual [s]")
    ax.set_title("Average time to find the first counterfactual per search algorithm")
    ax.set_xticks(x + width, [x.replace("Perturber", "").replace("Tokenizer", "") for x in [*x_axis, "Recompute", "Don't Recompute"]])
    # ax.set_xticks(x + width, x_axis)

    plt.savefig(results_path + "time_for_first_cf.png")

    # plt.show()

    plt.close(fig)
    # Search duration per input token for all Classifiers

    x = np.arange(len(x_axis))
    width = .25
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    for s, data in search_algos.items():
        d = dict()
        d[s] = []
        for a in x_axis:
            d[s].append(round(data[a]["duration_per_token"] / data[a]["total_input_token_entries"], 2))

        offset = width * multiplier
        rects = plt.bar(
            x + offset,
            d[s],
            width,
            label=s
        )
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_yscale("log")
    ax.legend()
    ax.set_ylabel("Average duration per input token [s]")
    ax.set_title("Average search duration per input token for all Classifiers")
    # ax.set_xticks(x + width, [x.replace("Perturber", "").replace("Tokenizer", "") for x in x_axis])
    ax.set_xticks(x + width, x_axis)
    # plt.show()
    plt.savefig(results_path + "time_per_token.png")

    plt.close(fig)

    # Search duration per counterfactual for all Classifiers

    x = np.arange(len(x_axis))
    width = .25
    multiplier = 0
    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    for s, data in search_algos.items():
        d = dict()
        d[s] = []
        for a in x_axis:
            d[s].append(round(data[a]["duration_per_cf"] / data[a]["total_cf_entries"], 2))

        offset = width * multiplier
        rects = plt.bar(
            x + offset,
            d[s],
            width,
            label=s.replace("_", " ").replace("Classifier", "").strip()
        )
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_yscale("log")
    ax.legend()
    ax.set_ylabel("Average search duration per counterfactual [s]")
    ax.set_title("Average search duration per counterfactual for all Classifiers")
    # ax.set_xticks(x + width, [x.replace("Perturber", "").replace("Tokenizer", "") for x in x_axis])
    ax.set_xticks(x + width, x_axis)
    # plt.show()
    plt.savefig(results_path + "time_per_cf.png")

    plt.close(fig)

    # Average number of counterfactuals per input token

    x = np.arange(len(x_axis))
    width = .25
    multiplier = 0
    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    for s, data in search_algos.items():
        d = dict()
        d[s] = []
        for a in x_axis:
            d[s].append(round((data[a]["total_cfs"] / data[a]["total_input_tokens"]), 2))

        offset = width * multiplier
        rects = plt.bar(
            x + offset,
            d[s],
            width,
            label=s.replace("_", " ").replace("Classifier", "").strip()
        )
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_yscale("log")
    ax.legend()
    ax.set_ylabel("Average number of counterfactuals per input token")
    ax.set_title("Average number of counterfactuals per input token")
    # ax.set_xticks(x + width, [x.replace("Perturber", "").replace("Tokenizer", "") for x in x_axis])
    ax.set_xticks(x + width, x_axis)
    # plt.show()
    plt.savefig(results_path + "number_of_cf.png")

    plt.close(fig)
    plt.clf()

    # Average number of counterfactuals per input token per model

    x = np.arange(len(classifiers))
    width = .25
    multiplier = 0
    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    for s, data in search_algos.items():
        d = dict()
        d[s] = []
        for a in classifiers:
            if data[a]["total_input_tokens"] == 0:
                d[s].append(0)
            else:
                d[s].append(round((data[a]["total_cfs"] / data[a]["total_input_tokens"]), 2))

        offset = width * multiplier
        rects = plt.bar(
            x + offset,
            d[s],
            width,
            label=s.replace("_", "-").replace("Classifier", "").strip().strip("-")
        )
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_yscale("log")
    ax.legend()
    ax.set_ylabel("Average number of counterfactuals per input token per model")
    ax.set_title("Average number of counterfactuals per input token per model")
    ax.set_xticks(x + width, [x.replace("Classifier", "").replace("_", "-").strip().strip("-") for x in classifiers])
    # plt.show()
    plt.savefig(results_path + "number_of_cf_per_model.png")

    plt.close(fig)
    plt.clf()

    # Average search duration per token per model

    x = np.arange(len(classifiers))
    width = .25
    multiplier = 0
    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    for s, data in search_algos.items():
        d = dict()
        d[s] = []
        for a in classifiers:
            if data[a]["total_input_token_entries"] == 0:
                d[s].append(0)
            else:
                d[s].append(round((data[a]["duration_per_token"] / data[a]["total_input_token_entries"]), 2))

        offset = width * multiplier
        rects = plt.bar(
            x + offset,
            d[s],
            width,
            label=s.replace("_", "-").replace("Classifier", "").strip().strip("-")
        )
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_yscale("log")
    ax.legend(loc="right")
    ax.set_ylabel("Average search duration per input token [s]")
    ax.set_title("Average search duration per input token per model of code")
    ax.set_xticks(x + width, [x.replace("Classifier", "").replace("_", "-").strip().strip("-") for x in classifiers])
    # plt.show()
    plt.savefig(results_path + "duration_per_token_per_model.png")

    plt.close(fig)
    plt.clf()

    # Average search duration per cf per model

    x = np.arange(len(classifiers))
    width = .25
    multiplier = 0
    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    for s, data in search_algos.items():
        d = dict()
        d[s] = []
        for a in classifiers:
            if data[a]["total_cf_entries"] == 0:
                d[s].append(0)
            else:
                d[s].append(round((data[a]["duration_per_cf"] / data[a]["total_cf_entries"]), 2))

        offset = width * multiplier
        rects = plt.bar(
            x + offset,
            d[s],
            width,
            label=s.replace("_", "-").replace("Classifier", "").strip().strip("-")
        )
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_yscale("log")
    ax.legend()
    ax.set_ylabel("Average search duration per counterfactual [s]")
    ax.set_title("Average search duration per counterfactual per model of code")
    ax.set_xticks(x + width, [x.replace("Classifier", "").replace("_", "-").strip().strip("-") for x in classifiers])
    # plt.show()
    plt.savefig(results_path + "duration_per_cf_per_model.png")

    plt.close(fig)
    plt.clf()

    # average num of cf per search run for all search alogs, including lig
    colors = ["steelblue", "darkorange", "forestgreen", "red"]
    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    total_x = []
    total_y = []

    for s, data in search_algos.items():
        total_x.append(s)
        runs = data["total_entries"]
        if runs == 0:
            total_y.append(0)
        else:
            total_y.append(round((data["total_cfs"] / runs), 2))

    total_x.append("LigSearch")
    runs = lig["total_entries"]
    if runs == 0:
        total_y.append(0)
    else:
        total_y.append(round((lig["total_cfs"] / runs), 2))

    x = np.arange(len(total_x))

    rects = plt.bar(total_x, total_y, color=colors)
    ax.bar_label(rects, padding=3)

    ax.set_yscale("log")
    ax.set_ylabel("Average number of counterfactuals")
    ax.set_title("Average number of counterfactuals per search run for all search algorithms")
    ax.set_xticks(x, total_x)
    plt.savefig(results_path + "cf_per_run.png")
    # plt.show()

    plt.close(fig)
    plt.clf()

    # average duration for a cf for all search alogs, including lig

    fig, ax = plt.subplots(layout="constrained", figsize=(22 * cm, 12 * cm))

    total_x = []
    total_y = []

    for s, data in search_algos.items():
        total_x.append(s)
        runs = data["total_cfs"]
        if runs == 0:
            total_y.append(0)
        else:
            total_y.append(round((data["total_duration"] / runs), 2))

    total_x.append("LigSearch")
    runs = lig["total_cfs"]
    if runs == 0:
        total_y.append(0)
    else:
        total_y.append(round((lig["total_duration"] / runs), 2))

    x = np.arange(len(total_x))

    rects = plt.bar(total_x, total_y, color=colors)
    ax.bar_label(rects, padding=3)

    ax.set_yscale("log")
    ax.set_ylabel("Average search duration per counterfactual [s]")
    ax.set_title("Average search duration per counterfactual for all search algorithms")
    ax.set_xticks(x, total_x)
    plt.savefig(results_path + "time_per_search_algo.png")
    # plt.show()

    plt.close(fig)
    plt.clf()


def print_invalid_classifications(totals, invalids):
    endl = " \\\\\\hline\n"
    table_str = "Classifier & Total source code samples & Invalid classifications & Percentage of total [%]" + endl

    names = sorted(totals.keys())

    for classifier in names:
        total_num_of_classifications = totals[classifier]
        num = invalids[classifier]
        table_str += classifier.replace("Classifier", "").replace("MLP_", "MLP") + " & " + str(total_num_of_classifications) + " & " + str(num) + " & " + str(round((num / total_num_of_classifications) * 100, 2)) + endl

    print(
        """
\\begin{table}[h]
   \\centering
   \\small
   \\begin{tabularx}{\\textwidth}{X|X|X|X}\n""" +
        table_str.replace("_", "\\_").replace("%", "\\%") + """
   \\end{tabularx}
   \\caption{
   The numbers of total classifications and invalid classifications for each classifier. 
   An invalid classification is when the model identifies a source code sample as invulnerable when it is labeled as vulnerable in the CodeXGLUE test dataset.
   In this case the search for counterfactuals is aborted before it is started.
   The fourth column gives the relative percentage with respect to the total number of evaluated source code snippets.
   }
   \\label{tab:invalids}
\\end{table}
    """.strip("\n"))


def print_removed_types(results, removed, names):
    current_index = 1
    indices = dict()  # [name] = col index
    lines = [["Configuration"]]

    for params in results.keys():
        line = ["-"] * len(indices)
        line.insert(0, names[params])
        lines.append(line)

        total_removed = 0
        for (t, number) in removed[params].items():
            total_removed += number

        for (t, number) in removed[params].items():
            if t + " [%]" in lines[0]:
                index = indices[t]
            else:
                lines[0].append(t + " [%]")
                indices[t] = current_index
                index = current_index
                current_index += 1
                line.append("-")
            if index != 1:
                pass
            line[index] = str(round((number / total_removed) * 100, 2))

    for line in lines:
        if len(line) < current_index:
            for i in range(len(line), current_index):
                line.append("-")

    line_end = " \\\\\\hline\n"
    table_str = ""
    for line in lines:
        table_str += " & ".join(line) + line_end
    print(
        """
\\begin{table}[h]
   \\centering
   \\small
   \\begin{tabularx}{15cm}{""" + "|".join(["X" for i in lines[0]]) + "}\n" +
        table_str.replace("_", "\\_").replace("%", "\\%") + """
   \\end{tabularx}
   \\caption{
   The percentage of counterfactuals with $\\leq$ 2 changes to the input with removals containing the types of program statements in the table.
   Categories of removed tokens are explained in Section \\ref{sec:removed_token_classification}.
   }
   \\label{tab:removals}
\\end{table}
    """.strip("\n"))


def print_parameters(results, names):
    endl = " \\\\\\hline\n"

    indices = dict()  # [column header] = index
    current_index = 1  # 0 = abbreviations
    table = [["Abbreviation"]]

    set = list(results.keys())
    set = sorted(set, key=lambda a: a.unmasker)
    set = sorted(set, key=lambda a: a.tokenizer)
    set = sorted(set, key=lambda a: a.perturber)
    set = sorted(set, key=lambda a: a.search_algorithm)
    set = sorted(set, key=lambda a: a.classifier)

    for r in set:
        line = ["n.a."] * len(indices)
        line.insert(0, names[r])

        for (p_name, value) in r.parameters.items():
            name = p_name.capitalize().replace("_", " ")
            if name == "Allow syntax errors in counterfactuals":
                # always true
                continue

            if name not in indices:
                indices[name] = current_index
                index = current_index
                current_index += 1
                line.append("n.a.")
                table[0].append(name)
            else:
                index = indices[name]

            if type(value) == float:
                line[index] = str(round(value, 2))
            else:
                line[index] = str(value)

        table.append(line)

    for line in table:
        if len(line) < current_index:
            for i in range(len(line), current_index):
                line.append("n.a.")

    line_end = " \\\\\\hline\n"
    table_str = ""
    for line in table:
        table_str += " & ".join(line) + line_end
    print(
        """
\\begin{table}[h]
   \\centering
   \\small
   \\hspace*{-4.5cm}\\begin{tabularx}{28cm}{""" + "|".join(["X" for i in table[0]]) + "}\n" +
        table_str + """
   \\end{tabularx}
   \\caption{Search parameters for every combination of search configuration. 
   Not every parameter is applicable to every configuration.
    Inapplicable parameters are denoted as "n.a.".}
   \\label{tab:parameters}
\\end{table}
    """.strip("\n"))


def get_and_print_names(results):
    names = dict()

    endl = " \\\\\\hline\n"
    table = "Abbreviation&Classifier & Search Algorithm & Perturber & Tokenizer & Unmasker" + endl

    set = list(results.keys())
    set = sorted(set, key=lambda a: a.unmasker)
    set = sorted(set, key=lambda a: a.tokenizer)
    set = sorted(set, key=lambda a: a.perturber)
    set = sorted(set, key=lambda a: a.search_algorithm)
    set = sorted(set, key=lambda a: a.classifier)

    for p in set:
        classifier = p.classifier
        search_algorithm = p.search_algorithm
        perturber = p.perturber
        tokenizer = p.tokenizer
        unmasker = p.unmasker

        classifier = classifier.replace("Classifier", "")
        search_algorithm = search_algorithm.replace("SearchAlgorithm", "")
        perturber = perturber.replace("Perturber", "")
        tokenizer = tokenizer.replace("Tokenizer", "")
        unmasker = unmasker.replace("Unmasker", "")

        abbreviation = ""
        abbreviation += "" if classifier == "NotApplicable" else "".join(re.findall("([A-Z0-9])", classifier)) + " "
        abbreviation += "Gr " if search_algorithm == "Greedy" else "GA " if search_algorithm == "Genetic" else "" if search_algorithm == "NotApplicable" else "".join(re.findall("([A-Z])", search_algorithm)) + " "
        abbreviation += "" if perturber == "NotApplicable" else "".join(re.findall("([A-Z0-9])", perturber)) + " "
        abbreviation += "" if tokenizer == "NotApplicable" else "".join(re.findall("([A-Z0-9])", tokenizer)) + " "
        abbreviation += "" if unmasker == "NotApplicable" else "".join(re.findall("([A-Z0-9])", unmasker)) + " "

        abbreviation = abbreviation.replace("VBERTMLP", "VB")

        table += abbreviation + " & " + \
                 ("n.a." if classifier == "NotApplicable" else classifier.replace("VulBERTa_MLP_", "VulBERTa")) + " & " + \
                 ("n.a." if search_algorithm == "NotApplicable" else search_algorithm.replace("KExpExhaustiveSearch", "$k$EES")) + " & " + \
                 ("n.a." if perturber == "NotApplicable" else perturber) + " & " + \
                 ("n.a." if tokenizer == "NotApplicable" else tokenizer) + " & " + \
                 ("n.a." if unmasker == "NotApplicable" else unmasker) + endl

        names[p] = abbreviation

    print("""
\\begin{table}[h]
   \\centering
   \\small
   \\begin{tabularx}{15cm}{l|l|l|l|l|l}""".strip()
          )
    print(table.replace("_", "\\_").replace("%", "\\%"), end="")
    print("""
   \\end{tabularx}
   \\caption{The abbreviations for all combinations of search configurations.}
   \\label{tab:abbreviations}
\\end{table}""".strip("\n")
          )

    return names


def print_table(results, names, times, numbers_of_cf, absolute_small_changes, relative_small_changes, similarities):
    table_list = [["Configuration", "Avg. processing time per token [ms]", "Avg. number of CFs per token", "Total number of CFs", "Number of CFs with $\\leq$ 2 changes", "Number of CFs with $\\leq$ 20% of input tokens changed", "Avg. string similarity of input to CF"]]

    for a in results.keys():
        line = [
            names[a],
            str(round(times[a]["average_per_token"] * 1000, 2)),
            str(round(numbers_of_cf[a]["average_number_of_cf"], 2)),
            str(numbers_of_cf[a]["total_number_of_cf"]),
            str(len(absolute_small_changes[a])),
            str(len(relative_small_changes[a])),
            str(round(similarities[a], 4))
        ]
        table_list.append(line)

    line_end = " \\\\\\hline\n"
    table = ""
    for line in table_list:
        table += " & ".join(line) + line_end

    print("""
\\begin{table}[h]
   \\centering
   \\small
   \\hspace*{-4.5cm}\\begin{tabularx}{28cm}""".strip() +
          "{" +
          "X|" * len(table_list[0]) +  # + 1 for left most label column
          "X}"
          )
    print(table.replace("_", "\\_").replace("%", "\\%"), end="")
    print("""
   \\end{tabularx}
   \\caption{The results from every combination of search algorithm, tokenizer, perturber, unmasker and classifier.
   String similarity is calculated using the Jaro distance \\cite{basak_computing_2023}, where 1 means identical strings, and 0 means no similarity between two strings.}   
   \\label{tab:results}
\\end{table}""".strip("\n")
          )


def plot_times(times, names):
    y = []
    x_per_token = []
    for (params, info) in times.items():
        y.append(names[params])
        x_per_token.append(info["average_per_token"])
    fig, ax = plt.subplots()
    ax.bar(y, x_per_token)
    ax.set_xlabel("Search parameters")
    ax.set_ylabel("Time [s]")
    ax.set_title("Average time per token")
    ax.set_yscale("log")
    ax.set_xlim(left=0.5)
    ax.tick_params(axis='x', labelrotation=-45)
    plt.savefig(results_path + "times.png")
    print(results_path + "times.png")
    # plt.show()
    plt.close(fig)


def get_removed_types_on_small_change_cf(small_changes):
    result = dict()  # [params] = ([type] = number)
    for (params, cfs) in small_changes.items():
        temp = dict()  # [type] = number
        for cf in cfs:
            for (token, type_str) in cf["changed_tokens"].items():
                type = type_str.split(":")[0]
                temp[type] = temp.get(type, 0) + 1
        result[params] = temp
    return result


def plot_histogram_how_many_cf_with_how_many_changes(results, names):
    histogram = dict()
    max_category = 0
    print("plot_histogram_how_many_cf_with_how_many_changes")
    for (params, data) in results.items():
        try:
            specific_histogram = dict()
            specific_max_category = 0
            for d in data:
                for cf in d.counterfactuals:
                    num_changes = cf["number_of_changes"]
                    input_length = cf["number_of_tokens_in_input"]
                    relative = num_changes / input_length

                    category = int(relative * 100)
                    if category > max_category:
                        max_category = category
                    if category > specific_max_category:
                        specific_max_category = max_category

                    histogram[category] = histogram.get(category, 0) + 1
                    specific_histogram[category] = specific_histogram.get(category, 0) + 1
            # fig, ax = plt.subplots()
            # ax.bar([i for i in range(specific_max_category + 1)], [0 if i == 0 else specific_histogram.get(i, 0.1) for i in range(specific_max_category + 1)])
            # ax.set_xlabel("% of changes relative to input size")
            # ax.set_ylabel("Number of CFs")
            # ax.set_title("Distribution of relative CF sizes for " + names[params])
            # ax.set_ylim(bottom=0)
            # ax.set_xlim(left=0.5, right=specific_max_category + 5)
            ##ax.set_yscale('log')
            # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            # plt.savefig(results_path + names[params] + "_cf_size_dist.png")
            # print(results_path + names[params] + "_cf_size_dist.png")
            # plt.close(fig)
        except Exception as e:
            print(e)
    fig, ax = plt.subplots()
    ax.bar([i for i in range(max_category + 1)], [0 if i == 0 else histogram.get(i, 0) for i in range(max_category + 1)])
    ax.set_xlabel("% of changes relative to input size")
    ax.set_ylabel("Number of CFs")
    ax.set_title("Distribution of relative CF sizes over all search configurations")
    ax.set_ylim(bottom=.1)
    ax.set_xlim(left=-.25, right=max_category + 5)
    ax.set_yscale('log')
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(results_path + "combined_cf_size_dist.png")
    print(results_path + "combined_cf_size_dist.png")


def time_per_token(results):
    times = dict()
    for (params, data) in results.items():
        time_sum = 0.0
        time_per_token_sum = 0.0
        for d in data:
            time_sum += d.search_duration
            time_per_token_sum += d.search_duration / d.input_token_length
        time_avg = time_sum / len(data)
        time_per_token_avg = time_per_token_sum / len(data)
        times[params] = {"total_average": time_avg, "average_per_token": time_per_token_avg}
    return times


def get_cf_with_absolute_little_changes(results, names, cutoff: int = 2):
    littles = dict()
    for (params, data) in results.items():
        cfs = []
        for d in data:
            for cf in d.counterfactuals:
                if len(cf["changed_tokens"]) <= cutoff:
                    cfs.append(cf)
        littles[params] = cfs
    return littles


def get_cf_with_relative_little_changes(results, relative_cutoff: float = .2):
    littles = dict()
    for (params, data) in results.items():
        cfs = []
        for d in data:
            for cf in d.counterfactuals:
                if len(cf["changed_tokens"]) <= relative_cutoff * d.input_token_length:
                    cfs.append(cf)
        littles[params] = cfs
    return littles


def get_average_similarity(results, inputs):
    sims = dict()
    for (params, data) in results.items():
        similarity = 0.0
        dl = len(data)
        if dl == 0:
            sims[params] = 0.0
        for d in data:
            l = len(d.counterfactuals)
            if l == 0:
                continue
            str = d.get_input(inputs)
            cf_sim = 0.0
            for cf in d.counterfactuals:
                changed_src = cf["code"]
                sim = jellyfish.jaro_similarity(str, changed_src)
                cf_sim += sim

            similarity += cf_sim / l
        sims[params] = similarity / dl
    return sims


def number_of_cf(results):
    numbers = dict()
    for (params, data) in results.items():
        s = 0
        for d in data:
            s += len(d.counterfactuals)
        numbers[params] = {"total_number_of_cf": s, "average_number_of_cf": s / len(data)}
    return numbers


from os import listdir
from os.path import isfile, join

mypath = "D:\\A_Uni\\A_MasterThesis\\Results"

onlyfiles = [mypath + "\\" + f for f in listdir(mypath) if isfile(join(mypath, f))]
# onlyfiles = onlyfiles[:10]

print(onlyfiles)

eval(onlyfiles)
