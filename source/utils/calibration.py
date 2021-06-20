import numpy as np
from matplotlib import pyplot as plt
import os


def calculate_binned_prob_acc_list(probability_list, match_list, bins_limits):
    """

    :param probability_list: array of the max class prediction probability of a sample
    :param match_list: array with two values {0,1} which indicates whether the sample was predicted correctly {1}
    or incorrectly {0}
    :param bins_limits: list of the limits between the bins
    :return:
        - binned_probability_vec_test: mean class probability of each bin
        - binned_accuracy_vec_test: mean accuracy of each bin
    """
    binned_probability_list = np.zeros(np.shape(bins_limits)[0] + 1)
    binned_accuracy_list = np.zeros(np.shape(bins_limits)[0] + 1)
    binplace = np.digitize(probability_list, bins_limits, right=True)  # Returns which bin an array element belongs to

    for bin_num in range(0, np.shape(bins_limits)[0] + 1):
        if bin_num in binplace:
            binned_probability_list[bin_num] = np.mean(np.array(probability_list)[np.where(binplace == bin_num)])
            binned_accuracy_list[bin_num] = np.mean(np.array(match_list)[np.where(binplace == bin_num)])

    return binned_probability_list, binned_accuracy_list


# Expected Calibration Error - ECE
def calculate_expected_calibration_error(binned_probability_vec, binned_accuracy_vec, probability_list, bins_limits):
    frequency_in_bins_list = np.zeros(np.shape(bins_limits)[0] + 1)
    binplace = np.digitize(probability_list, bins_limits, right=True)  # Returns which bin an array element belongs to

    for bin_num in range(0, np.shape(bins_limits)[0] + 1):
        if bin_num in binplace:
            frequency_in_bins_list[bin_num] = np.shape(np.array(probability_list)[np.where(binplace == bin_num)])[0]
    ece = np.sum((frequency_in_bins_list / np.sum(frequency_in_bins_list)) * np.absolute(
        np.array(binned_accuracy_vec) - np.array(binned_probability_vec)))

    return ece


# Positive Expected Calibration Error - ECE
def calculate_pos_neg_expected_calibration_error(binned_probability_vec, binned_accuracy_vec, probability_list,
                                                 bins_limits):
    frequency_in_bins_list = np.zeros(np.shape(bins_limits)[0] + 1)
    binplace = np.digitize(probability_list, bins_limits, right=True)  # Returns which bin an array element belongs to

    for bin_num in range(0, np.shape(bins_limits)[0] + 1):

        if bin_num in binplace:
            frequency_in_bins_list[bin_num] = np.shape(np.array(probability_list)[np.where(binplace == bin_num)])[0]

    ece_positive = np.sum((frequency_in_bins_list / np.sum(frequency_in_bins_list)) * np.clip(
        np.array(binned_accuracy_vec) - np.array(binned_probability_vec), a_min=0, a_max=None))

    ece_negative = np.sum((frequency_in_bins_list / np.sum(frequency_in_bins_list)) * np.clip(
        np.array(binned_accuracy_vec) - np.array(binned_probability_vec), a_min=None, a_max=0))

    return ece_positive, ece_negative


# Maximum Calibration Error
def calculate_maximum_clibration_error(binned_probability_vec, binned_accuracy_vec):
    return np.max(np.absolute(np.array(binned_accuracy_vec) - np.array(binned_probability_vec)))


# save calibration bar chart as .png and save values in .txt files
def calibration_bar_chart(
        binned_probability_vec,
        binned_accuracy_vec,
        # savefolder,
        # filename,
        title='Calibration-Chart',
        xlabel='confidence',
        ylabel='accuracy',
        write_to_txtfile_Bool=True
):
    f1 = plt.figure()  # figsize=(6,6))
    plt.margins(0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    n_bins = np.shape(binned_probability_vec)[0]

    plt.xticks(np.arange(11), ([str(format(i, '.1f')) for i in list(np.arange(0, 1.1, 0.1))]))

    x_axis = [str(format(i, '.1f')) for i in list(np.arange(0, 1, 0.1))]

    optimal_accuracy = binned_probability_vec

    plt.plot(np.array(range(0, n_bins + 1)), np.array(range(0, n_bins + 1)) / 10, color='dimgray', linestyle='-.')

    outputs = plt.bar(x_axis, binned_probability_vec, width=1, align='edge', color='peachpuff', edgecolor='indianred',
                      label='Gap')
    gap = plt.bar(x_axis[::-1], binned_accuracy_vec, width=1, align='edge', color='steelblue', edgecolor='darkblue',
                  label='Outputs')
    plt.legend(handles=[outputs, gap])
    plt.show()
    plt.close()


def frequency_over_calibration(
        probability_list,
        bins_limits,
        # savefolder,
        # filename,
        scale_param=1,
        title='Frequency Over Confidence',
        xlabel='confidence',
        ylabel='frequency',
        write_to_txtfile_Bool=True
):
    """

    :param probability_list: array of the max class prediction probability of a sample
    :param bins_limits: list of the limits between the bins
    :param scale_param:
    :param title:
    :param xlabel:
    :param ylabel:
    :param write_to_txtfile_Bool:
    :return:
        prints histogram of frequency in bins
    """
    frequency_in_bins_list = np.zeros(np.shape(bins_limits)[0] + 1)
    binplace = np.digitize(probability_list, bins_limits, right=True)  # Returns which bin an array element belongs to

    for bin_num in range(0, np.shape(bins_limits)[0] + 1):
        if bin_num in binplace:
            frequency_in_bins_list[bin_num] = np.shape(np.array(probability_list)[np.where(binplace == bin_num)])[0]
    frequency_in_bins_list = frequency_in_bins_list / scale_param  # rescale array

    f1 = plt.figure()  # figsize=(6,6))
    plt.margins(0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    n_bins = np.shape(frequency_in_bins_list)[0]
    plt.xticks(np.arange(11), ([str(format(i, '.1f')) for i in list(np.arange(0, 1.1, 0.1))]))
    x_axis = [str(format(i, '.1f')) for i in list(np.arange(0, 1, 0.1))]
    plt.bar(x_axis, frequency_in_bins_list, width=1, align='edge', color='yellowgreen', edgecolor='olive')
    # f1.savefig(savefolder + filename + '.png', dpi=200)
    plt.show()
    plt.close()


# Input: class_probability_list:
# Ouput:
def class_probabilities(
        class_probability_list,
        # savefolder,
        # filename,
        title='Class Probabilities',
        xlabel='class',
        ylabel='mean probability',
        write_to_txtfile_Bool=True
):
    """

    :param class_probability_list: class prediction probability of samples (2D-array with shape (number_of_samples,number_of_classes))
    :param title:
    :param xlabel:
    :param ylabel:
    :param write_to_txtfile_Bool:
    :return:
        print histogram of sorted mean probabilities of each class
    """
    # sort class probability list from highest predicted probability to lowest

    class_probability_list = np.mean(np.sort(class_probability_list)[:, ::-1], axis=0)
    f1 = plt.figure()  # figsize=(6,6))
    plt.margins(0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    n_bins = np.shape(class_probability_list)[0]
    x_axis = [str(format(i, '.1f')) for i in list(np.arange(0, 10, 1))]
    plt.yticks(np.arange(0., 1.0, 0.2))
    plt.ylim(0, 1.)
    plt.bar(x_axis, class_probability_list, width=1, color='forestgreen', edgecolor='darkgreen')
    plt.show()
    plt.close()
