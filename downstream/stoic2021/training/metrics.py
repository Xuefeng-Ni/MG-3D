from sklearn.metrics import accuracy_score, roc_auc_score
from torch import sigmoid
from numpy import round

def acc_probcovid(preds, labels):
    preds = sigmoid(preds[:, 0]).numpy()
    labels = labels[:, 0].numpy()
    return accuracy_score(labels, round(preds))

def acc_probseverecovid(preds, labels):
    mask = labels[:, 0].numpy()
    preds = sigmoid(preds[:, 1]).numpy()
    preds = preds[mask > 0.5]
    labels = labels[:, 1].numpy()
    labels = labels[mask > 0.5]
    return accuracy_score(labels, round(preds))

def roc_probcovid(preds, labels):
    preds = preds[:, 0].numpy()
    labels = labels[:, 0].numpy()
    return roc_auc_score(labels, preds)

def roc_probseverecovid(preds, labels):
    mask = labels[:, 0].numpy()
    preds = preds[:, 1].numpy()
    preds = preds[mask > 0.5]
    labels = labels[:, 1].numpy()
    labels = labels[mask > 0.5]
    return roc_auc_score(labels, preds)