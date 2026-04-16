from enum import Enum

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
        return self.name

    def __repr__(self):
        return self.name

class LabelHandler:
    def __init__(self, labels: list[Label]):
        self.raw_labels = labels
        self.labels = []
        for label in labels:
            label = Label(label["label"], LabelType(label["category"]))
            self.labels.append(label)

    def getQueriableLabels(self):
        labels = []
        for label in self.labels:
            labels.append(label.name)
        return labels

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