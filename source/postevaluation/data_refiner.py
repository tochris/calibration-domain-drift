import numpy as np
import time
import sys

sys.path.append("..")

from source.perturbation_generator import PerturbationGenerator
from source.utils.plotutils import save_dict_to_txt
from source.utils.optimizer_nelder_mead import minimize_neldermead


def get_accuracy(gauss_eps, *args):
    """
    Calculates accuracy for a given level of perturbation (=epsilon) based on
    the dataset provided through dataset_name. Applies gaussian perturbation
    to the samples in the dataset according to the given level of perturbation.
    The accuracy is calculated from the perturbed datset and the resulting
    predictions of the model.

    Args:
        epsilon: float, level of gaussian perturbation
        (*args) modelf: object from class ModelFactory
        (*args) data: tf.data.Dataset object (x_data, y_labels), prepared with
            batch_size, shuffle etc.
        (*args) dataset_name: string
    Return:
        accuracy: float
    """

    modelf, data, dataset_name = args
    gauss_eps = gauss_eps
    perturb_generator = PerturbationGenerator(
        dataset_name=dataset_name,
        gaussian_noise_eps=[gauss_eps]
    )
    logits, labels = modelf.logits(
        data,
        perturb_generator,
        "general_gaussian_noise",
        range(1),
        from_cache=False,
        to_cache=False,
        save_to_file=False
    )
    pred_classes = np.argmax(logits, axis=1)
    labels = np.argmax(labels, axis=1)
    matches = np.equal(pred_classes, labels)
    accuracy = np.sum(matches) / np.shape(matches)[0]
    print("Accuracy: %s  Epsilon: %s" % (accuracy, gauss_eps))
    return accuracy


def optmize_accuracy(epsilons, *args):
    """
    Function that is used in the nelder mead optimizer.
    Args:
        epsilon: float, level of gaussian perturbation
        (*args) target_accuracy: float
        (*args) modelf: object from class ModelFactory
        (*args) data: tf.data.Dataset object (x_data, y_labels), prepared with
            batch_size, shuffle etc.
        (*args) dataset_name: string
    Return:
        float, difference between target accuracy and calculated accuracy
    """
    target_accuracy, modelf, data, dataset_name = args
    return abs(target_accuracy - get_accuracy(gauss_eps, modelf, data, dataset_name))


def estimate_epsilons(
        modelf,
        data,
        dataset_name,
        n_classes,
        number_perturbation_levels,
        accuracy_deviation_acceptable,
        accuracy_deviation_acceptable_last_step,
        gauss_eps_start=0.05,
        opt_delta_gauss_eps=0.5):
    """
    Optimizes levels of gaussian perturbation (=epsilons) based on a list
    of target accuracies.
    The target accuracies are calcualted with interpolation between min & max.
    The data is perturbed with gaussian noise at a certain level (=epsilon).
    The model's accuracy for the perturbed data is calculated.
    The levels of perturbation are iteratively optimized with nelder mead.

    Args:
        modelf: object from class ModelFactory
        data: tf.data.Dataset object (x_data, y_labels), prepared with
            batch_size, shuffle etc.
        dataset_name: string
        n_classes: int, number of classes in dataset
        number_perturbation_levels: int, number of intended levels of perturbation
        accuracy_deviation_acceptable: float, rate of acceptable deviation between
            target accuracy and calcualted accuracy based on level of perturbation
        accuracy_deviation_acceptable_last_step: float, rate of acceptable deviation
            between target accuracy and calcualted accuracy based on level of
            perturbation for the lowest accuracy
        gauss_eps_start: float, starting value for level of perturbation in optimization
    !    opt_delta_gauss_eps: float ????

    Return:
        optimized_epsilons: list [float], optimized levels of perturbation (=epsilons)
            according to the list of target accuracies.
    """

    start_time = time.time()
    # calculate target accuracies
    max_acc = get_accuracy(0.0, modelf, data, dataset_name)
    min_acc = 1 / n_classes
    perturbation_levels = range(number_perturbation_levels)
    target_accuracy_list = []
    for perturbation_level in perturbation_levels:
        target_accuracy_list.append(max_acc - (max_acc - min_acc) * perturbation_level / (len(perturbation_levels) - 1))
    # estimate gauss epsilons with regard to target accuracies
    print("Start estimating gaussian epsilons with regard to target accuracies: ", target_accuracy_list)
    epsilons = []
    for i, target_accuracy in enumerate(target_accuracy_list):
        print("Step_%s - Target Accuracy: %s" % (str(i), str(target_accuracy)))
        if i == 0:
            epsilons.append(0.0)
        else:
            if i == 1:
                x0 = gauss_eps_start
            else:
                x0 = epsilons[i - 1]

            if i == len(target_accuracy_list) - 1:
                tolerance = max_acc * accuracy_deviation_acceptable_last_step
            else:
                tolerance = max_acc * accuracy_deviation_acceptable

            x = minimize_neldermead(
                optmize_accuracy,
                x0,
                args=(target_accuracy, modelf, data, dataset_name),
                tol_abs=tolerance,
                nonzdelt=opt_delta_gauss_eps,
                callback=None,
                maxiter=None, maxfev=None, disp=False,
                return_all=False, initial_simplex=None,
                xatol=1e-4, fatol=1e-4, adaptive=False,
            )
            epsilons.append(x[0])
    time_needed = time.time() - start_time
    print("Finished estimating gaussian epsilons!!")
    print("Time needed: ", time_needed)
    return epsilons


def store_epsilons(
        model_path,
        epsilons,
        modelf,
        data,
        data_name,
        filename="optimized_epsilons"):
    """
    Stores levels of gaussian perturbation (=epsilons) to a file.
    Args:
        model_path: string,
        epsilons: list [float]
        modelf: object from class ModelFactory
        data: tf.data.Dataset object (x_data, y_labels), prepared with
            batch_size, shuffle etc.
        dataset_name: string
        filename: string, e.g."optimized_epsilons"
    """

    resulting_accuracy_list = []
    for gauss_eps in epsilons:
        resulting_accuracy_list.append(get_accuracy(gauss_eps, modelf, data, data_name))
    dict_gauss_results = {"Epsilons of Gaussian Perturbation:   ": epsilons,
                          "Accuracies of Gaussian Perturbation: ": resulting_accuracy_list}
    print("Epsilons: ", epsilons)
    print("Accuracies: ", resulting_accuracy_list)
    print("File path: ", model_path)
    print()
    save_dict_to_txt(model_path, dict_gauss_results, filename)
