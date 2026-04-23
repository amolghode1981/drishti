from fileinput import filename
from os import read

from networkx import dfs_edges, draw
from numpy import average, var
from pandas import DataFrame
from pygments.lexer import combined
from rich import json
from sympy import plot

from src.web.labels import LabelHandler
from src.web.db import get_collection, list_images
import numpy as np


class Analysis:
    def __init__(self, checksum):
        self.checksum = checksum
        self.collection = get_collection()
        self.doc = self.collection.find_one({"_id": checksum})
        if self.doc is None:
            raise ValueError(f"Image with checksum {checksum} not found in database.")
        self.label_handler = LabelHandler([])
        self.readings = self.doc.get("readings", [])

    def __map_labels(self, readings):
        for reading in readings:
            labels = reading.get("labels", [])
            self.label_handler.addLabels(labels)

    def __debug_print(self,dict: dict | DataFrame):
        import json
        print(json.dumps(dict, indent=2))

    def __process_readings(self, readings):
        # Process readings to compute average scores for each label
        self.__map_labels(readings)
        label_scores = {}
        label_count = {}

        # Aggregate scores for each label across all readings
        for reading in readings:
            for match in reading.get("matches", []):
                label = match["label"]
                score = match["score"]
                if label not in label_scores:
                    label_scores[label] = []
                label_scores[label].append(score)
                if label not in label_count:
                    label_count[label] = 0
                label_count[label] += 1

        # Compute the sum of all the labels
        label_sum = {label: sum(scores) for label, scores in label_scores.items()}
        # Compute the average score for each label and update the LabelHandler
        for label, count in label_count.items():
            average_score = label_sum[label] / count
            self.label_handler.setScore(label, average_score)

    def analyzeImage(self):
        try:
            self.__process_readings(self.readings)
            # self.__debug_print({"labels": self.label_handler.getAsDataframe().to_dict(orient="records")})
            musical_labels = self.label_handler.getMusicalLabels()
            non_musical_labels = self.label_handler.getNonMusicalLabels()
            self.__debug_print({"musical_labels": [str(label) for label in musical_labels], "non_musical_labels": [str(label) for label in non_musical_labels]})
            # musical_score = 0
            # for label in musical_labels:
            #     musical_score += label.score
            # print(f"Musical score average  : {musical_score/len(musical_labels) if musical_labels else 0:.4f}")
            # compute_non_musical_score = 0
            # for label in non_musical_labels:
            #     compute_non_musical_score += label.score
            # print(f"Non-musical score average  : {compute_non_musical_score/len(non_musical_labels) if non_musical_labels else 0:.4f}")

            compute_musical_score = average([label.score for label in musical_labels]) if musical_labels else 0
            compute_non_musical_score = average([label.score for label in non_musical_labels]) if non_musical_labels else 0
            #print(f"Non-musical labels: {non_musical_labels}")
            musical_to_non_musical_ratio = (compute_musical_score / compute_non_musical_score)
            output = {
                "musical_score": compute_musical_score,
                "non_musical_score": compute_non_musical_score,
                "outcome": "MUSICAL" if musical_to_non_musical_ratio > 1 else "NON-MUSICAL" if musical_to_non_musical_ratio < 1 else "INCONCLUSIVE",
            }
            # if musical_to_non_musical_ratio > 1:
            #     print(f"Image {self.checksum[:12]}... is likely MUSICAL (ratio: {musical_to_non_musical_ratio:.2f})")
            # elif musical_to_non_musical_ratio < 1:
            #     print(f"Image {self.checksum[:12]}... is likely NON-MUSICAL (ratio: {musical_to_non_musical_ratio:.2f})")
            return output
        except Exception as e:
            print(f"Error analyzing image {self.checksum[:12]}...: {e}")
            return {
                "musical_score": 0,
                "non_musical_score": 0,
                "outcome": "ERROR",
            }
    '''
    Doesnt seem to be of any significance to compute the variance of the scores, as it is not a measure of confidence but rather a measure of consistency across readings.
    A high variance could indicate that the model is uncertain about the label, while a low variance could indicate that the model is confident about the label.
      However, in practice, the variance may not be a reliable indicator of confidence,
      as it can be influenced by various factors such as the quality of the image, the diversity of the training data,
      and the complexity of the model. Therefore, it may be more effective to focus on the average scores and their ratios to
      determine whether an image is musical or non-musical, rather than relying on variance as a measure of confidence.
    '''
    # def analyzeImageCombinedSignal(self):
    #     try:
    #         self.__process_readings(self.readings)
    #         # self.__debug_print({"labels": self.label_handler.getAsDataframe().to_dict(orient="records")})
    #         musical_labels = self.label_handler.getMusicalLabels()
    #         non_musical_labels = self.label_handler.getNonMusicalLabels()

    #         musical_score = 0
    #         # for label in musical_labels:
    #         #     musical_score += label.score
    #         # print(f"Musical score average  : {musical_score/len(musical_labels) if musical_labels else 0:.4f}")
    #         # compute_non_musical_score = 0
    #         # for label in non_musical_labels:
    #         #     compute_non_musical_score += label.score
    #         # print(f"Non-musical score average  : {compute_non_musical_score/len(non_musical_labels) if non_musical_labels else 0:.4f}")

    #         compute_musical_score = average([label.score for label in musical_labels]) if musical_labels else 0
    #         compute_non_musical_score = average([label.score for label in non_musical_labels]) if non_musical_labels else 0
    #         variance_musical_score = var([label.score for label in musical_labels]) if musical_labels else 0
    #         variance_non_musical_score = var([label.score for label in non_musical_labels]) if non_musical_labels else 0
    #         print(f"variance musical score: {variance_musical_score:.6f}, variance non-musical score: {variance_non_musical_score:.6f} ")
    #         print(f"Non-musical labels: {non_musical_labels}")
    #         musical_signal = compute_musical_score / (variance_musical_score + 1e-6)
    #         non_musical_signal = compute_non_musical_score / (variance_non_musical_score + 1e-6)
    #         musical_to_non_musical_ratio = (musical_signal / non_musical_signal) if non_musical_signal > 0 else float('inf')
    #         output = {
    #             "musical_score": musical_signal,
    #             "non_musical_score": non_musical_signal,
    #             "outcome": "MUSICAL" if musical_to_non_musical_ratio > 1 else "NON-MUSICAL" if musical_to_non_musical_ratio < 1 else "INCONCLUSIVE",
    #         }
    #         # if musical_to_non_musical_ratio > 1:
    #         #     print(f"Image {self.checksum[:12]}... is likely MUSICAL (ratio: {musical_to_non_musical_ratio:.2f})")
    #         # elif musical_to_non_musical_ratio < 1:
    #         #     print(f"Image {self.checksum[:12]}... is likely NON-MUSICAL (ratio: {musical_to_non_musical_ratio:.2f})")
    #         return output
    #     except Exception as e:
    #         print(f"Error analyzing image {self.checksum[:12]}...: {e}")
    #         return {
    #             "musical_score": 0,
    #             "non_musical_score": 0,
    #             "outcome": "ERROR",
    #         }

    def permutation_test(self, num_permutations=1000):

        # This should set the average scores for each label in the LabelHandler
        self.__process_readings(self.readings)
        filename =  self.doc.get("filename", "unknown")
        # Perform a permutation test to assess the statistical significance of the musical vs non-musical classification
        musical_labels = self.label_handler.getMusicalLabels()
        non_musical_labels = self.label_handler.getNonMusicalLabels()
        observed_ratio = (average([label.score for label in musical_labels]) / average([label.score for label in non_musical_labels])) if non_musical_labels else float('inf')

        combined_scores = [label.score for label in musical_labels + non_musical_labels]
        #self.__debug_print({"combined_scores": combined_scores, "observed_ratio": observed_ratio})
        count_extreme = 0
        permuted_ratios = []
        for _ in range(num_permutations):
            shuffled = combined_scores.copy()         # fresh copy each iteration
            np.random.shuffle(shuffled)
            permuted_musical_scores = shuffled[:len(musical_labels)]
            permuted_non_musical_scores = shuffled[len(musical_labels):]
            permuted_ratio = (average(permuted_musical_scores) / average(permuted_non_musical_scores)) if average(permuted_non_musical_scores) > 0 else float('inf')
            permuted_ratios.append(permuted_ratio)
            if permuted_ratio >= observed_ratio:
                count_extreme += 1

        p_value = count_extreme / num_permutations
        plot_permutation_distribution(permuted_ratios, observed_ratio, p_value, filename)
        return p_value, observed_ratio

import matplotlib.pyplot as plt

def plot_permutation_distribution(permuted_ratios, observed_ratio, p_value, filename):
    """
    permuted_ratios : list of floats — all 1000 random ratios
    observed_ratio  : float — your real ratio
    p_value         : float — computed p-value
    filename        : str — name of the file being analyzed
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # All random ratios
    ax.hist(permuted_ratios, bins=40, color='#888782',
            edgecolor='#444441', linewidth=0.5,
            label='random shuffle ratios')

    # Ratios that beat your observed
    extreme = [r for r in permuted_ratios if r >= observed_ratio]
    if extreme:
        ax.hist(extreme, bins=40, color='#E24B4A',
                edgecolor='#501313', linewidth=0.5,
                label=f'beats observed ({len(extreme)}/{len(permuted_ratios)})')

    # Your observed ratio
    ax.axvline(x=observed_ratio, color='#5DCAA5',
               linewidth=2, linestyle='--',
               label=f'observed ratio = {observed_ratio:.3f}')

    ax.set_xlabel('ratio (musical mean / non-musical mean)')
    ax.set_ylabel('count')
    ax.set_title(f'{filename}  |  p = {p_value:.4f}  |  ratio = {observed_ratio:.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig('permutation_distribution.png', dpi=150)
    plt.show()

def interpret_p_value(p_value: float, observed_ratio: float, num_permutations: int = 1000) -> dict:
    beats_you = round(p_value * num_permutations)
    confidence = round((1 - p_value) * 100, 1)

    if p_value < 0.05 and observed_ratio > 1:
        verdict     = "Musical"
        explanation = (f"{beats_you}/{num_permutations} random shuffles beat your ratio. "
                    f"Musical labels are dominating. Confident this is a musical image.")

    elif p_value > 0.95 and observed_ratio < 1:
        verdict     = "Non-Musical"
        explanation = (f"{beats_you}/{num_permutations} random shuffles beat your ratio. "
                    f"Non-musical labels are dominating. Confident this is a non-musical image.")

    elif p_value < 0.05 and observed_ratio < 1:
        verdict     = "Non-Musical (weak)"
        explanation = (f"{beats_you}/{num_permutations} random shuffles beat your ratio. "
                    f"Ratio is below 1 — non-musical labels winning but signal is weak.")

    elif p_value > 0.95 and observed_ratio > 1:
        verdict     = "Suspicious"
        explanation = (f"{beats_you}/{num_permutations} random shuffles beat your ratio. "
                    f"Ratio is above 1 but almost all shuffles beat you — check your label set.")
    else:
        verdict     = "Ambiguous"
        explanation = (f"{beats_you}/{num_permutations} random shuffles beat your ratio. "
                    f"Too many random groupings perform as well — no confident signal.")

    return {
        "verdict":          verdict,
        "p_value":          p_value,
        "observed_ratio":   round(observed_ratio, 4),
        "beats_you":        beats_you,
        "confidence_pct":   confidence,
        "explanation":      explanation,
    }

if __name__ == "__main__":
    docs = list_images()
    p_values = []
    for d in docs:
        id = str(d["_id"])
        analysis = Analysis(id)
        p_value, observed_ratio = analysis.permutation_test()
        p_values.append({
            "id": id,
            "p_value": p_value,
            "observed_ratio": observed_ratio,
            "filename": d["filename"],
        })
        #print(f"id: {id}, p-value: {p_value:.4f}")

    for res in p_values:
        result = interpret_p_value(res["p_value"], res["observed_ratio"])
        # if result["verdict"] == "Ambiguous":
        #     print(f"Image {res['id']}...: {result['verdict']} (p-value: {result['p_value']:.4f}, observed ratio: {result['observed_ratio']:.4f}) - {result['explanation']}")
        print(f"Image {res['filename']} ID: {res['id']}: {result['verdict']} (p-value: {result['p_value']:.4f}, observed ratio: {result['observed_ratio']:.4f})")