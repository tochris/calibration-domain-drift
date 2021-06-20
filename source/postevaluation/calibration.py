import numpy as np
from matplotlib import pyplot as plt
import os


def calculate_binned_prob_acc_list(probs, matches, bins_limits):
    """Calculate binned probabilities and accuracies 
    :param probs: [float], array of max probabilities
    :param matches: [int], array with two values {0,1} which indicates, whether
        the sample was predicted correctly {1} or incorrectly {0}
    :param bins_limits: [float], list of boarders between bins
    return:
        binned_probs: [float], mean class probability of each bin
        bins_limits: [float], mean accuracy of each bin
    """
    binned_probs = np.zeros(np.shape(bins_limits)[0] + 1)
    binned_accs = np.zeros(np.shape(bins_limits)[0] + 1)
    # Returns which bin an array element belongs to
    binplace = np.digitize(probs, bins_limits, right=True)
    for bin_num in range(0, np.shape(bins_limits)[0] + 1):
        if bin_num in binplace:
            to_replace = np.where(binplace == bin_num)
            binned_probs[bin_num] = np.mean(np.array(probs)[to_replace])
            binned_accs[bin_num] = np.mean(np.array(matches)[to_replace])
    return binned_probs, binned_accs


def calculate_expected_calibration_error(binned_probability_vec, binned_accuracy_vec, probability_list, bins_limits):
    """Calculate Expected Calibration Error
    :param binned_probability_vec: [float], mean class probability of each bin
    :param binned_accuracy_vec: [float], mean accuracy of each bin
    :param probability_list: [float], array of max probabilities
    :param bins_limits: [float], list of boarders between bins
    return:
        ECE: int
    """
    frequency_in_bins_list = np.zeros(np.shape(bins_limits)[0] + 1)
    binplace = np.digitize(probability_list, bins_limits, right=True)  # Returns which bin an array element belongs to
    for bin_num in range(0, np.shape(bins_limits)[0] + 1):
        if bin_num in binplace:
            frequency_in_bins_list[bin_num] = np.shape(np.array(probability_list)[np.where(binplace == bin_num)])[0]
    ece = np.sum((frequency_in_bins_list / np.sum(frequency_in_bins_list)) * np.absolute(
        np.array(binned_accuracy_vec) - np.array(binned_probability_vec)))
    return ece
