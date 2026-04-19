from os import read

from networkx import dfs_edges
from numpy import average, var
from pandas import DataFrame
from rich import json

from src.web.labels import LabelHandler
from src.web.db import get_collection
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
            print(f"Non-musical labels: {non_musical_labels}")
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
