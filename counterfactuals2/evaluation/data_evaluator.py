import json
import matplotlib
from matplotlib.ticker import MaxNLocator
import jellyfish
import re

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from counterfactuals2.evaluation.EvaluationResult import EvaluationResult, EvaluationParameters, EvaluationData

results_path = "results/"


def eval(path):
    with open(path) as file:
        content = file.read()
    data = json.loads(content)
    duration = data["duration_sec"]
    inputs = data["ids_of_input"]
    raw_results = data["results"]
    print("duration", duration)
    results = dict()
    for raw in raw_results:
        eval_result = EvaluationResult(raw)
        params = EvaluationParameters(eval_result)
        data = EvaluationData(eval_result)
        results.setdefault(params, []).append(data)

    times = time_per_token(results)
    numbers_of_cf = number_of_cf(results)
    absolute_small_changes = get_cf_with_absolute_little_changes(results)
    relative_small_changes = get_cf_with_relative_little_changes(results, .2)
    removed = get_removed_types_on_small_change_cf(absolute_small_changes)
    similarities = get_average_similarity(results, inputs)

    print(times)
    print(numbers_of_cf)
    print(absolute_small_changes)
    print(relative_small_changes)
    print(removed)
    print(similarities)

    # plot_histogram_how_many_cf_with_how_many_changes(results,names)
    # plot_times(times,names)

    names = get_and_print_names(results)
    print("\n")
    print_table(results, names, times, numbers_of_cf, absolute_small_changes, relative_small_changes, similarities)
    print("\n")
    print_parameters(results, names)
    print("\n")
    print_removed_types(results, removed, names)


def print_removed_types(results, removed, names):
    current_index = 1
    indices = dict()  # [name] = col index
    lines = [["Configuration"]]

    for params in results.keys():
        line = ["-"] * len(indices)
        line.insert(0, names[params] + " [%]")
        lines.append(line)

        for (t, number) in removed[params].items():
            if t + " [%]" in lines[0]:
                index = indices[t]
            else:
                lines[0].append(t)
                indices[t] = current_index
                index = current_index
                current_index += 1
                line.append("-")
            if index != 1:
                pass
            line[index] = str(round((number / len(removed[params])) * 100, 2))

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
   \\caption{Search parameters for every combination of search configuration.}
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

        table += abbreviation + " & " + \
                 ("n.a." if classifier == "NotApplicable" else classifier) + " & " + \
                 ("n.a." if search_algorithm == "NotApplicable" else search_algorithm) + " & " + \
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
          "X|" * len(results.keys()) +  # + 1 for left most label column
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
        fig, ax = plt.subplots()
        ax.bar([i for i in range(max_category + 1)], [0 if i == 0 else histogram.get(i, 0.01) for i in range(max_category + 1)])
        ax.set_xlabel("% of changes relative to input size")
        ax.set_ylabel("Number of CFs")
        ax.set_title("Distribution of relative CF sizes for " + names[params])
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(results_path + names[params] + ".png")
        print(results_path + names[params] + ".png")
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar([i for i in range(max_category + 1)], [0 if i == 0 else histogram.get(i, 0.01) for i in range(max_category + 1)])
    ax.set_xlabel("% of changes relative to input size")
    ax.set_ylabel("Number of CFs")
    ax.set_title("Distribution of relative CF sizes over all search configurations")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(results_path + "combined.png")
    print(results_path + "combined.png")


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


def get_cf_with_absolute_little_changes(results, cutoff: int = 2):
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


eval("D:\\A_Uni\\A_MasterThesis\\MMD\\counterfactuals2/json_dump_4.17.0_2024_June_03__10_55_57.json")
