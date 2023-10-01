from sklearn.metrics import balanced_accuracy_score

def predict_score(classifier, data, labels):
    lab_pred = classifier.predict(data)
    score = balanced_accuracy_score(labels, lab_pred)
    return score