import calibration
import numpy as np
from scipy.special import softmax
from sklearn.metrics import brier_score_loss, log_loss

import source.postevaluation.calibration as calib


def accuracy(logits, labels):
    """
    Calculate the accuracy of given logits and labels.
    Softmax transformation happens internally.
    :param logits: Float array, outer list = samples, inner list = logits
    :param labels: Float array, outer list = samples, inner list = 1 hot labels
    :return: [Float] of length 1
    """

    pred = np.argmax(logits, axis=1)
    truth = np.argmax(labels, axis=1)
    match = np.equal(pred, truth)

    return [np.mean(match)]

def brier_score(logits, labels):
    """
    Calculate brier score of two NxD matrices.
    N sample size, D dimension size, P probality matrix, L label matrix.
    General formula:
    brier_score =
        1 / N *
        sum for d = 1...D of
        sum for n = 1...N of
        (P_nd - L_nd)^2
    :param logits: Float array, outer list = samples, inner list = logits
    :param labels: Float array, outer list = samples, inner list = 1 hot labels
    :return: [Float] of length 1
    """
    # transform logits to probabilities and clip values for numerical stability
    probs = softmax(logits, axis=1)
    probs = np.clip(probs, 1.17e-37, 3.40e37)
    # transpose and form label-prob pairs in each dimension
    lp_pairs = zip(np.transpose(labels), np.transpose(probs))
    # calculate brier score in each dimension
    brier_scores = [brier_score_loss(label, prob) for label, prob in lp_pairs]
    # sum over all dimensions
    return [np.sum(brier_scores)]

def neg_log_likelihood(logits, labels):
    """
    Calculate negative log likelihood loss based on logits and labels.
    For this, logits are transformed to probabilities.
    :param logits: Float array, outer list = samples, inner list = logits
    :param labels: Float array, outer list = samples, inner list = 1 hot labels
    :return: [Float] of length 1
    """
    # transform logits to probabilities and clip values for numerical stability
    probs = softmax(logits, axis=1)
    probs = np.clip(probs, 1.17e-37, 3.40e37)
    return [log_loss(labels, probs)]


def ECE(logits, labels):
    """
    Calculate the ECE of given logits and labels.
    Softmax transformation happens internally.
    :param logits: Float array, outer list = samples, inner list = logits
    :param labels: Float array, outer list = samples, inner list = 1 hot labels
    :return: [Float] of length 1
    """

    def probability_accuracy_binned(
            logits,
            labels,
            bins_calibration=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ):
        """
        Evaluation metric.
        Cannot use this for tuning since it doesn't return a single float.
        Use this for plotting.
        :param logits: Float array, outer list = samples, inner list = logits
        :param labels: Float array, outer list = samples, inner list = 1 hot labels
        :param bins_calibration: Float array, list of boundaries between the bins
        :return: ([Float], [Float])
        """

        pred = np.argmax(logits, axis=1)
        truth = np.argmax(labels, axis=1)
        match = np.equal(pred, truth)

        probs = softmax(logits, axis=1)
        probs = np.clip(probs, 1.17e-37, 3.40e37)
        max_prob = np.ndarray.max(probs, axis=1)

        binned_prob, binned_acc = calib.calculate_binned_prob_acc_list(
            max_prob, match, bins_calibration
        )

        return binned_prob, binned_acc

    bins_calibration = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # transform logits to probabilities and clip values for numerical stability
    probs = softmax(logits, axis=1)
    probs = np.clip(probs, 1.17e-37, 3.40e37)
    # get the highest predicted probability in each sample
    max_prob = np.ndarray.max(probs, axis=1)
    # calculate binned confidence and accuracy values
    binned_prob, binned_acc = probability_accuracy_binned(
        logits, labels, bins_calibration
    )
    # calculate step ECE and accumulate
    ece = calib.calculate_expected_calibration_error(
        binned_prob,
        binned_acc,
        max_prob,
        bins_calibration,
    )
    return [ece]

def vuc_ECE(logits, labels):
    """
    Calculate the CE (Verified Uncertainty Paper) of given logits and labels.
    :param logits: Float array, outer list = samples, inner list = logits
    :param labels: Float array, outer list = samples, inner list = 1 hot labels
    :return: [Float] of length 1
    """

    # transform logits to probabilities and clip values for numerical stability
    probs = softmax(logits, axis=1)
    probs = np.clip(probs, 1.17e-37, 3.40e37)
    labels = np.argmax(labels, axis=1)

    calibration_error = calibration.get_calibration_error(probs, labels)

    return [calibration_error]

def confidence_scores(logits, labels=None):
    """
    Evaluation metric.
    Cannot use this for tuning since it doesn't return a single float.
    Use this for plotting.
    :param logits: Float array, outer list = samples, inner list = logits
    :param labels: Float array, outer list = samples, inner list = 1 hot labels
    :return: [Float]
    """

    probs = softmax(logits, axis=1)
    probs = np.clip(probs, 1.17e-37, 3.40e37)
    return np.max(probs, axis=1)

def matches(logits, labels):
    """
    Evaluation metric.
    Cannot use this for tuning since it doesn't return a single float.
    Use this for plotting/micro-averaging.
    :param logits: Float array, outer list = samples, inner list = logits
    :param labels: Float array, outer list = samples, inner list = 1 hot labels
    :return: [Float]
    """

    pred = np.argmax(logits, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.equal(pred, truth)

def mean_entropy(logits, labels=None):
    """
    Calculate entropy based on logits.
    For this, logits are transformed to probabilities.
    The argument `labels` is unused, but kept for consistency with other
    measures.
    Formula (N samples, D dimensions, p NxD probability matrix):
    S = 1/N *
        sum i=1...N of
        sum j=1...D of
        - p_ij * log(p_ij)
    :param logits: Float array, outer list = samples, inner list = logits
    :param labels: Float array, outer list = samples, inner list = 1 hot labels
    :return: [Float] of length 1
    """
    # transform logits to probabilities and clip values for numerical stability
    probs = softmax(logits, axis=1)
    probs = np.clip(probs, 1.17e-37, 3.40e37)
    entrops = [-np.sum(np.multiply(prob, np.log(prob))) for prob in probs]
    return [np.mean(entrops)]
