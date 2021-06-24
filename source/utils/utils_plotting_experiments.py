import numpy as np
import pandas as pd
import os
import sys
import calibration as cal

os.chdir("..")
sys.path.append(os.getcwd())
from source.postevaluation.evaluator import Evaluator
from source.utils.calibration import calculate_expected_calibration_error,\
                                     calculate_binned_prob_acc_list


def get_data(data_model, losses):
    """Load pandas dataframe with results of model evaluation"""
    dump = [
    ]
    # extend paths to storage files
    exp_results_path = [
        path
        for path, _, _, _ in data_model
    ]
    # create dataframe with all evaluation data
    evaluators = [Evaluator() for _ in exp_results_path]

    for evaluator, path, dm in zip(evaluators, exp_results_path, data_model):
        evaluator.load(path)
        evaluator.storage["NN"] = dm[3]
        evaluator.storage["Model"] = dm[2]
        evaluator.storage["Data"] = dm[1]
        evaluator.storage["Data_Model_NN"] = dm[1] + " " + dm[2] + " " + dm[3]
        evaluator.storage = evaluator.storage.loc[
            evaluator.storage.Loss_type.isin(losses)
        ]
    data_all = pd.concat([evaluator.storage for evaluator in evaluators])

    return data_all

def process_micro_vucECE(df):
    """Calculate micro vuc ECE for each perturbation"""
    df_list = []
    for dm in df.Data_Model_NN.unique():
        df_list.append(micro_avg_ece(dm, df,vucECE=True))
    df_microVUCECE = pd.concat(df_list, axis=0,ignore_index=True)
    return df_microVUCECE

def process_micro_ECE(df):
    """Calculate micro ECE for each perturbation"""
    df_list = []
    for dm in df.Data_Model_NN.unique():
        df_list.append(micro_avg_ece(dm, df,vucECE=False))
    df_microECE = pd.concat(df_list, axis=0,ignore_index=True)
    return df_microECE

def micro_avg_ece_helper(identity, df, pert, vucECE=False):
    """
    Calculates calibration error
    requires "confidence_scores" and "matches" in  df
    """
    bins_limits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    df = df.loc[df.Data_Model_NN==identity]
    df = df[df.Perturbation==pert]
    binned_prob, binned_acc = calculate_binned_prob_acc_list(
        df[df.Loss_type=="confidence_scores"].Value,
        df[df.Loss_type=="matches"].Value,
        bins_limits
    )

    if vucECE == False:
        return {pert: calculate_expected_calibration_error(
            binned_prob,
            binned_acc,
            df[df.Loss_type=="confidence_scores"].Value,
            bins_limits
        )}
    else:
        return {pert: cal.get_calibration_error(
            df[df.Loss_type=="confidence_scores"].Value,
            df[df.Loss_type=="matches"].Value.astype(int),
        )}

def micro_avg_ece(Data_Model_NN, data_all, vucECE=False):
    """Calculate average micro ece"""
    li = [micro_avg_ece_helper(Data_Model_NN, data_all, pert, vucECE) for pert \
          in set(data_all.Perturbation)]
    dict_all = {}
    for li_ in li:
        dict_all[list(li_.keys())[0]] = list(li_.values())[0]
    df = pd.DataFrame.from_dict(dict_all, orient="index", columns=["Value"])
    df["Perturbation"] = df.index.values
    df["Model"] = data_all.Model[data_all.Data_Model_NN==Data_Model_NN].unique()[0]
    return df

def clean_df(df):
    """Clean dataframe from perturbations and models as well as rename perturbations"""
    df = df.loc[df.Model.isin(["Base-P"])==False]
    df = df.loc[df.Perturbation.isin(["general_gaussian_noise"])==False]
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_shot_noise", "Perturbation"] = "shot noise"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_impulse_noise", "Perturbation"] = "impulse noise"
    df.loc[df["Perturbation"]==  \
        "imagenet2012_corrupted_defocus_blur", "Perturbation"] = "defocus blur"
    df.loc[df["Perturbation"]==  \
        "imagenet2012_corrupted_glass_blur", "Perturbation"] = "glass blur"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_motion_blur", "Perturbation"] = "motion blur"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_zoom_blur", "Perturbation"] = "zoom blur"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_snow", "Perturbation"] = "snow"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_frost", "Perturbation"] = "frost"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_contrast", "Perturbation"] = "contrast"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_elastic_transform", "Perturbation"] = "elastic transform"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_pixelate", "Perturbation"] = "pixelate"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_gaussian_blur", "Perturbation"] = "gaussian blur"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_spatter", "Perturbation"] = "spatter"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_speckle_noise", "Perturbation"] = "speckle noise"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_fog", "Perturbation"] = "fog"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_brightness", "Perturbation"] = "brightness"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_jpeg_compression", "Perturbation"] = "jpeg compr"
    df.loc[df["Perturbation"]== \
        "imagenet2012_corrupted_saturate", "Perturbation"] = "saturate"
    return df
