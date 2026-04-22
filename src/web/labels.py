from enum import Enum

import pandas

class LabelType(Enum):
    MUSICAL = "Musical"
    NON_MUSICAL = "Non-Musical"
    GENERIC = "Generic"

class Label:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.score = 0.0

    def __str__(self):
        return f"<Label: {self.name}, score: {self.score:.4f}>"

    def __repr__(self):
        return self.name

class LabelHandler:
    def __init__(self, labels: list[Label]):
        self.raw_labels = labels
        self.labels = []
        for label in labels:
            label = Label(label["label"], LabelType(label["category"]))
            self.labels.append(label)

    def addLabel(self, label: dict):
        if not self.hasLabel(label["label"]):
            label = Label(label["label"], LabelType(label["category"]))
            self.labels.append(label)
            self.raw_labels.append(label)

    def addLabels(self, labels: list[dict]):
        for label in labels:
            if not self.hasLabel(label["label"]):
                label = Label(label["label"], LabelType(label["category"]))
                self.labels.append(label)
                self.raw_labels.append(label)

    def getQueriableLabels(self):
        labels = []
        for label in self.labels:
            labels.append(label.name)
        return labels

    def hasLabel(self, label_name):
        for label in self.labels:
            if label.name == label_name:
                return True
        return False

    def getMusicalLabels(self):
        return [label for label in self.labels if label.type == LabelType.MUSICAL]
    def getNonMusicalLabels(self):
        return [label for label in self.labels if label.type == LabelType.NON_MUSICAL]
    def getGenericLabels(self):
        return [label for label in self.labels if label.type == LabelType.GENERIC]
    def setScore(self, label_name, score):
        for label in self.labels:
            if label.name == label_name:
                label.score = score
                break
    def getScores(self):
        return {label.name: label.score for label in self.labels}
    def getAsDataframe(self):
        import pandas as pd
        data = []
        for label in self.labels:
            data.append({
                "label": label.name,
                "score": label.score,
                "percentage": f"{label.score*100:.2f}%"
            })

        dataframe = pandas.DataFrame(data)
        dataframe = dataframe.sort_values(by="score", ascending=False).reset_index(drop=True)
        dataframe["score"] = dataframe["score"].apply(lambda x: round(float(x), 4))
        return dataframe
