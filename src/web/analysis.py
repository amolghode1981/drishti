from os import read

from networkx import dfs_edges
from numpy import average
from pandas import DataFrame
from rich import json

from src.web.labels import LabelHandler
from src.web.db import get_collection


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
        self.__process_readings(self.readings)
        self.__debug_print({"labels": self.label_handler.getAsDataframe().to_dict(orient="records")})
        musical_labels = self.label_handler.getMusicalLabels()
        non_musical_labels = self.label_handler.getNonMusicalLabels()
        generic_labels = self.label_handler.getGenericLabels()
        compute_musical_score = average([label.score for label in musical_labels]) if musical_labels else 0
        compute_non_musical_score = average([label.score for label in non_musical_labels]) if non_musical_labels else 0
        compute_generic_score = average([label.score for label in generic_labels]) if generic_labels else 0
        print(f"Musical Score: {compute_musical_score:.4f}")
        print(f"Non-Musical Score: {compute_non_musical_score:.4f}")
        print(f"Generic Score: {compute_generic_score:.4f}")