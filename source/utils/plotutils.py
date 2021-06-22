import matplotlib.pyplot as plt
import csv
import pickle
import os


def save_dict_to_structured_txt(savefolder, results_dict, filename):
    """
    Saves dictionary in a txt file where the items in dict are seperated by rows
    """
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    file_path = os.path.join(savefolder, filename + '.txt')
    file1 = open(file_path, 'w')
    for key in results_dict.keys():
        file1.write(str(key) + ':   ' + str(results_dict[key]) + '\n')
    file1.close()


def save_dict_to_txt(savefolder, dic, filename):
    """
    Saves a dictionary as txt file
    """
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    file_path = os.path.join(savefolder, filename + '.txt')
    f = open(file_path, 'w')
    f.write(str(dic))
    f.close()


def save_dict_to_csv(savefolder, params_dict, filename):
    """
    Saves dict as csv and pickled file
    """
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    # save params in csv file
    file_path = os.path.join(savefolder, filename + '.csv')
    writer = csv.writer(open(file_path, 'w'))
    for key, val in params_dict.items():
        writer.writerow([key, val])


def save_dict_to_pkl(savefolder, dic_logits, filename):
    """
    Saves dict as pickled file
    """
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    # save params in pkl file
    save_name = os.path.join(savefolder, filename + '.pkl')
    with open(save_name, "wb") as f:
        pickle.dump(dic_logits, f)
    f.close()


def load_dict_from_pkl(savefolder, filename):
    """
    Loads dict from pickled file
    """
    save_name = os.path.join(savefolder, filename + '.pkl')
    with open(save_name, 'rb') as f:
        params = pickle.load(f)
    f.close()
    return params
