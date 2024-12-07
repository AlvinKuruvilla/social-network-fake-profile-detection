import enum
import json
import logging
from multiprocessing import Pool

from sklearn.metrics import top_k_accuracy_score
import numpy as np
from tabulate import tabulate
from classifiers.template_generator import all_ids
from performance_evaluation.heatmap import HeatMap, VerifierType

original_beta = 0.68
original_r = 100000000


def print_k_table(matrix, ids, identifier):
    rows = []
    rows.append([1, top_k_accuracy_score(np.array(ids), np.array(matrix), k=1)])
    rows.append([2, top_k_accuracy_score(np.array(ids), np.array(matrix), k=2)])
    rows.append([3, top_k_accuracy_score(np.array(ids), np.array(matrix), k=3)])
    rows.append([4, top_k_accuracy_score(np.array(ids), np.array(matrix), k=4)])
    rows.append([5, top_k_accuracy_score(np.array(ids), np.array(matrix), k=5)])
    table = tabulate(rows, headers=["K", "Score"], tablefmt="plain")
    logging.info(identifier)
    logging.info(table)
    return list({row[0]: row[1] for row in rows}.values())


class ConfigKey(enum.Enum):
    R_VALUE = 1
    BETA = 2


def modify_key(key: ConfigKey, new_value: int | float):
    with open("classifier_config.json", "r") as file:
        config = json.load(file)

    # Modify the corresponding key based on the Enum value
    if key == ConfigKey.R_VALUE:
        config["dbod_r"] = new_value
    elif key == ConfigKey.BETA:
        config["dbod_beta"] = new_value

    # Write the updated configuration back to the JSON file
    with open("classifier_config.json", "w") as file:
        json.dump(config, file, indent=4)


def check_for_simple_majority(baseline_array, other_array):
    comparison = np.array(baseline_array) > np.array(other_array)

    # Count how many comparisons are True and return whether there is a mjority
    return (np.sum(comparison), np.sum(comparison) > len(baseline_array) / 2)


def compute_matrix(args):
    heatmap, pair, third, none1, none2, last = args
    return heatmap.combined_keystroke_matrix(pair, third, none1, none2, last)


def cross_platform_2v1():
    k_scores = []
    heatmap = HeatMap(VerifierType.ITAD)
    # Define the arguments for each matrix computation
    tasks = [
        (heatmap, [1, 2], 3, None, None, 1),
        (heatmap, [1, 3], 2, None, None, 1),
        (heatmap, [2, 1], 3, None, None, 1),
        (heatmap, [2, 3], 1, None, None, 1),
        (heatmap, [3, 1], 2, None, None, 1),
        (heatmap, [3, 2], 1, None, None, 1),
    ]

    # Parallel execution
    with Pool(processes=6) as pool:  # Use the number of CPU cores you have
        matrices = pool.map(compute_matrix, tasks)

    # Unpack the results
    matrix, matrix2, matrix3, matrix4, matrix5, matrix6 = matrices

    ids = all_ids()
    print()
    k_scores.extend(print_k_table(matrix=matrix, ids=ids, identifier="FI"))

    k_scores.extend(print_k_table(matrix=matrix2, ids=ids, identifier="FT"))

    k_scores.extend(print_k_table(matrix=matrix3, ids=ids, identifier="IF"))

    k_scores.extend(print_k_table(matrix=matrix4, ids=ids, identifier="IT"))

    k_scores.extend(print_k_table(matrix=matrix5, ids=ids, identifier="TF"))

    k_scores.extend(print_k_table(matrix=matrix6, ids=ids, identifier="TI"))
    return k_scores


if __name__ == "__main__":
    logging.basicConfig(filename="beta_test.log", level=logging.INFO)

    logging.info(
        f"Collecting baseline results with r: {original_r} and beta: {original_beta}"
    )
    baseline_kscores = cross_platform_2v1()
    for value in np.arange(0.3, 0.7, 0.1):
        logging.info(f"Setting beta to {value}")
        modify_key(ConfigKey.BETA, value)
        new_kscores = cross_platform_2v1()
        majority_value, majority_check = check_for_simple_majority(
            baseline_kscores, new_kscores
        )
        if majority_check:
            logging.info(f"Majority found at beta: {value}")
            break
    print("Resetting beta")
    modify_key(ConfigKey.BETA, original_beta)
