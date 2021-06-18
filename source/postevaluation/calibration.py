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
    :param binned_probability_vec_test: [float], mean class probability of each bin
    :param binned_accuracy_vec_test: [float], mean accuracy of each bin
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


# # Positive Expected Calibration Error - ECE
# def calculate_pos_neg_expected_calibration_error(binned_probability_vec, binned_accuracy_vec, probability_list,
#                                                  bins_limits):
#     frequency_in_bins_list = np.zeros(np.shape(bins_limits)[0] + 1)
#     binplace = np.digitize(probability_list, bins_limits, right=True)  # Returns which bin an array element belongs to
#     for bin_num in range(0, np.shape(bins_limits)[0] + 1):
#         if bin_num in binplace:
#             frequency_in_bins_list[bin_num] = np.shape(np.array(probability_list)[np.where(binplace == bin_num)])[0]
#     ece_positive = np.sum((frequency_in_bins_list / np.sum(frequency_in_bins_list)) * np.clip(
#         np.array(binned_accuracy_vec) - np.array(binned_probability_vec), a_min=0, a_max=None))
#     ece_negative = np.sum((frequency_in_bins_list / np.sum(frequency_in_bins_list)) * np.clip(
#         np.array(binned_accuracy_vec) - np.array(binned_probability_vec), a_min=None, a_max=0))
#     return ece_positive, ece_negative
#
#
# # Maximum Calibration Error
# def calculate_maximum_clibration_error(binned_probability_vec, binned_accuracy_vec):
#     return np.max(np.absolute(np.array(binned_accuracy_vec) - np.array(binned_probability_vec)))
#
#
# # save calibration bar chart as .png and save values in .txt files
# def save_calibration_bar_chart(binned_probability_vec, binned_accuracy_vec, savefolder, filename,
#                                title='Calibration-Chart', xlabel='confidence', ylabel='accuracy',
#                                write_to_txtfile_Bool=True):
#     # Make a folder for storing files of the model
#     if not os.path.exists(savefolder):
#         os.makedirs(savefolder)
#
#     f1 = plt.figure()  # figsize=(6,6))
#     plt.margins(0)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#
#     n_bins = np.shape(binned_probability_vec)[0]
#     plt.xticks(np.arange(11), ([str(format(i, '.1f')) for i in list(np.arange(0, 1.1, 0.1))]))
#     x_axis = [str(format(i, '.1f')) for i in list(np.arange(0, 1, 0.1))]
#     optimal_accuracy = binned_probability_vec
#
#     plt.plot(np.array(range(0, n_bins + 1)), np.array(range(0, n_bins + 1)) / 10, color='dimgray', linestyle='-.')
#     # plt.bar(x_axis, np.arange(0,1,0.1), width=1 ,align='edge', color='peachpuff', edgecolor='indianred')
#     outputs = plt.bar(x_axis, binned_probability_vec, width=1, align='edge', color='peachpuff', edgecolor='indianred',
#                       label='Gap')
#     gap = plt.bar(x_axis, binned_accuracy_vec, width=1, align='edge', color='steelblue', edgecolor='darkblue',
#                   label='Outputs')
#     plt.legend(handles=[outputs, gap])
#
#     f1.savefig(savefolder + filename + '.png', dpi=200)
#     plt.show()
#     plt.close()
#
#     ##Write lists to file
#     if write_to_txtfile_Bool == True:
#         file1 = open(savefolder + filename + '_binned_probability.txt', 'w')
#         for i in binned_probability_vec:
#             file1.write(str(i) + ', ')
#         file1.close()
#         file1 = open(savefolder + filename + '_binned_accuracy.txt', 'w')
#         for i in binned_accuracy_vec:
#             file1.write(str(i) + ', ')
#         file1.close()
#
#
# #
# #
# #
# ##Input: probability_vec_test_list: array of the max class prediction probability of a sample
# ##       bins_limits: list of the limits between the bins
# ##Ouput: print histogram of frequency in bins
# def save_frequency_over_calibration(probability_list, bins_limits, savefolder, filename, scale_param=1,
#                                     title='Frequency Over Confidence', xlabel='confidence', ylabel='frequency',
#                                     write_to_txtfile_Bool=True):
#     frequency_in_bins_list = np.zeros(np.shape(bins_limits)[0] + 1)
#     binplace = np.digitize(probability_list, bins_limits, right=True)  # Returns which bin an array element belongs to
#     for bin_num in range(0, np.shape(bins_limits)[0] + 1):
#         if bin_num in binplace:
#             frequency_in_bins_list[bin_num] = np.shape(np.array(probability_list)[np.where(binplace == bin_num)])[0]
#     frequency_in_bins_list = frequency_in_bins_list / scale_param  # rescale array
#
#     # Make a folder for storing files of the model
#     if not os.path.exists(savefolder):
#         os.makedirs(savefolder)
#
#     f1 = plt.figure()  # figsize=(6,6))
#     plt.margins(0)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#
#     n_bins = np.shape(frequency_in_bins_list)[0]
#     plt.xticks(np.arange(11), ([str(format(i, '.1f')) for i in list(np.arange(0, 1.1, 0.1))]))
#     x_axis = [str(format(i, '.1f')) for i in list(np.arange(0, 1, 0.1))]
#     plt.bar(x_axis, frequency_in_bins_list, width=1, align='edge', color='yellowgreen', edgecolor='olive')
#
#     f1.savefig(savefolder + filename + '.png', dpi=200)
#     plt.show()
#     plt.close()
#
#     ##Write lists to file
#     if write_to_txtfile_Bool == True:
#         file1 = open(savefolder + filename + '_frequency_over_calibration.txt', 'w')
#         for i in frequency_in_bins_list:
#             file1.write(str(i) + ', ')
#         file1.close()
#
#
# ##Input: class_probability_list: class prediction probability of samples (2D-array with shape (number_of_samples,number_of_classes))
# ##Ouput: print histogram of sorted mean probabilities of each class
# def save_class_probabilities(class_probability_list, savefolder, filename, title='Class Probabilities', xlabel='class',
#                              ylabel='mean probability', write_to_txtfile_Bool=True):
#     # print('SHAPEC:',np.shape(class_probability_list))
#     # sort class probability list from highest predicted probability to lowest
#     # 100x10 probs -> sort every sample -> get mean of every nth highest probability class
#     class_probability_list = np.mean(np.sort(class_probability_list)[:, ::-1], axis=0)
#
#     # Make a folder for storing files of the model
#     if not os.path.exists(savefolder):
#         os.makedirs(savefolder)
#
#     f1 = plt.figure()  # figsize=(6,6))
#     plt.margins(0)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#
#     n_bins = np.shape(class_probability_list)[0]
#     x_axis = [str(format(i, '.1f')) for i in list(np.arange(0, 10, 1))]
#     plt.yticks(np.arange(0., 1.0, 0.2))
#     plt.ylim(0, 1.)
#     plt.bar(x_axis, class_probability_list, width=1, color='forestgreen', edgecolor='darkgreen')
#
#     f1.savefig(savefolder + filename + '.png', dpi=200)
#     plt.show()
#     plt.close()
#
#     ##Write lists to file
#     if write_to_txtfile_Bool == True:
#         file1 = open(savefolder + filename + '_class_probabilities.txt', 'w')
#         for i in class_probability_list:
#             file1.write(str(i) + ', ')
#         file1.close()
