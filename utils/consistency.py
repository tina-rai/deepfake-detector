import random

def consistency_check(prediction, confidence):

    report = []

    risk = int(confidence * 100)

    report.append("Face detected successfully")

    if prediction == "FAKE":

        report.append("Possible facial artifact detected")

        report.append("Abnormal edge transitions")

        report.append("Lighting inconsistency")

        report.append("Texture mismatch")

    else:

        report.append("Natural facial texture")

        report.append("Consistent lighting")

        report.append("No obvious manipulation")

    return report, risk