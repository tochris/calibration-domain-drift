import numpy as np
import pickle
import os


def exp_id(x):
    """
    x |->
        exp(x), if x < 0
        x + 1,  otherwise
    """
    x = np.array(x)
    cond = x < 0
    return np.add(
        # circumvent possible overflow
        np.multiply(cond, np.exp(np.minimum(x, 1))),
        np.multiply(np.subtract(1, cond), np.add(x, 1))
    )


# deprecated
def load_loglab(file_directory):
    """
    Returns logits_test vector and labels_test vector in a given
    file_directory (each vector has shape: (nb_perturb_steps, nb_test_samples,
    nb_classes))
    """

    logits_file_name = 'logits_test.npy'
    labels_file_name = 'labels_test.npy'

    return \
        np.load(file_directory + logits_file_name), \
        np.load(file_directory + labels_file_name)


# deprecated
def save_evaluation(evaluation, file_directory):
    """
    Saves the evaluation array to a npy file in the given direcotry.
    """

    file_name = file_directory + '/evaluation.npy'
    np.save(file_name, evaluation)


def save_obj(obj, path, filename):
    # construct correct file path
    file_path = os.path.join(path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(path, filename):
    # construct correct file path
    file_path = os.path.join(path, filename)
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def save_obj_txt(obj, path, filename):
    # construct correct file path
    file_path = os.path.join(path, filename)
    with open(file_path, 'w') as f:
        f.write(str(obj))
        f.close()
