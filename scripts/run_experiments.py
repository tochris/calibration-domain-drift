import sys
import argparse
import tensorflow as tf

sys.path.append("..")

import source.data.cifar10 as cifar10
import source.data.imagenet as imagenet
import source.data.objectnet as objectnet
import source.postevaluation.data_refiner
from source.model_factory import ModelFactory
from source.PostCalibrator import PostCalibrator
from source.perturbation_generator import PerturbationGenerator

from source.postevaluation.measures import (
    accuracy,
    brier_score,
    neg_log_likelihood,
    ECE,
    vuc_ECE,
    confidence_scores,
    matches,
    mean_entropy,
)

from source.postevaluation.calib_model import (
    BaseModel,
    TSModel,
    ETSModel,
    IRMModel,
    IROVAModel,
    IROVATSModel,
)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run_experiment(settings):
    """
    run data refiner, calibrate model and evaluate model on data under domain drift
    Args:
        settings: dictionary with arguments
    """
    tuning_perturb = settings["tuning_perturb"]
    perturb_levels = settings["perturb_levels"]
    perturbations = settings["perturbations"]
    calib_models = settings["cms"]
    data_name = settings["data_name"]
    test_name = settings["test_name"]
    model_class = settings["model_class"]
    model_path = settings["model_path"]
    model_load = settings["model_load"]
    path_data_imagenet = settings["path_data_imagenet"]
    path_data_objectnet = settings["path_data_objectnet"]
    path_data_imagenet_corrupted = settings["path_data_imagenet_corrupted"]

    folder_path_save = settings["folder_path_save"]
    modelf, dataset_valid, dataset_test = load_modelf_dataset(
        model_class,
        model_path,
        data_name,
        model_load=model_load,
        path_data_imagenet=path_data_imagenet,
        path_data_objectnet=path_data_objectnet,
        path_data_imagenet_corrupted=path_data_imagenet_corrupted
    )

    if data_name in ["CIFAR"]:
        n_classes = 10
    elif data_name in ["Imagenet", "Objectnet_not_imagenet", "Objectnet_only_imagenet"]:
        n_classes = 1000

    if tuning_perturb:
        # calculate number of perturbation levels
        if perturb_levels == 1:
            tuning_eps = [0]
        elif perturb_levels > 1:
            tuning_eps = range(perturb_levels)
        else:
            ValueError: "perturb_levels must be > 0!"

        # adjust parameters for data sets and models
        # (depending on respective preprocessing of data)
        if data_name in ["Imagenet", "Objectnet_not_imagenet", "Objectnet_only_imagenet"]:
            if model_class in ["DenseNet169", "Xception", "MobileNetV2"]:
                gauss_eps_start = 0.0005
                opt_delta_gauss_eps = 0.5
            if model_class in ["ResNet50", "ResNet152", "VGG19", "EfficientNetB7"]:
                gauss_eps_start = 0.1
                opt_delta_gauss_eps = 0.5
        else:
            gauss_eps_start = 0.1
            opt_delta_gauss_eps = 0.5

        # optimize levels of perturbation (=epsilons)
        epsilons = data_refiner.estimate_epsilons(
            modelf,
            dataset_valid.valid_ds,
            data_name, n_classes,
            number_perturbation_levels=perturb_levels,
            accuracy_deviation_acceptable=0.03,
            accuracy_deviation_acceptable_last_step=0.05,
            gauss_eps_start=gauss_eps_start,
            opt_delta_gauss_eps=opt_delta_gauss_eps
        )

        # initialize perturbation generator with opimal levels of perturbation
        perturb_generator = PerturbationGenerator(
            dataset=dataset_test,
            dataset_name=data_name,
            gaussian_noise_eps=epsilons
        )

        # store levels of perturbation
        store_epsilons(
            model_path,
            epsilons, modelf,
            dataset_valid.valid_ds,
            data_name,
            filename="results_gaus_perturb"
        )

    else:
        # initialize perturbation generator without opimal levels of perturbation
        perturb_generator = PerturbationGenerator(
            dataset=dataset_test,
            dataset_name=data_name
        )

    for calib_model in calib_models:
        # create calibration instance
        pc = PostCalibrator(
            calib_model,
            modelf,
            perturb_generator,
            data_name,
            folder_path=folder_path_save
        )

        # tune model
        if calib_model.__class__.__name__ != "BaseModel":
            pc.tune(
                getattr(dataset_valid, "valid_ds"),
                tuning_perturb,
                tuning_eps,
                from_cache=False,
                to_cache=False,
                save_to_file=False
            )

        eval_metrics = [
            accuracy,
            ECE,
            vuc_ECE,
            brier_score,
            neg_log_likelihood,
            mean_entropy,
            confidence_scores,
            matches,
        ]

        # evaluate model
        for perturb in perturbations:
            n_eps = len(perturb_generator.possible_epsilons(perturb_type=perturb))
            print("perturbation: ", perturb)
            for epsilon in range(n_eps):
                for eval_metric in eval_metrics:
                    pc.evaluate(
                        getattr(dataset_test, test_name),
                        eval_metric,
                        perturb,
                        epsilon,
                        from_cache=True,
                        to_cache=True,
                        save_to_file=False,
                    )

        pc.save()

    if data_name not in ["Objectnet_not_imagenet", "Objectnet_only_imagenet"]:
        modelf.save_logits()


def load_modelf_dataset(model_class,
                        model_path,
                        data_name,
                        model_load,
                        path_data_imagenet=None,
                        path_data_objectnet=None,
                        path_data_imagenet_corrupted=None,
                        **kwarg):
    """
    Load model factory and dataset
    Args:
        model_class: string, e.g. ResNet50
        model_path: string, path where model is stored
        data_name: string, e.g. Imagenet
        model_load: bool, True if model should be loaded
    Return:
        modelf: object from class ModelFactory
        data_valid: tf.data.Dataset object (x_data, y_labels), prepared with
            batch_size, shuffle etc.
        data_test: tf.data.Dataset object (x_data, y_labels), prepared with
            batch_size, shuffle etc.
    """
    if data_name == "CIFAR":
        data_valid = data_test = cifar10.CIFAR10(train_batch_size=100)
    elif data_name == "Imagenet":
        data_valid = data_test = imagenet.Imagenet(data_path=path_data_imagenet,
                                                   model=model_class)
    elif data_name == "Objectnet_not_imagenet":
        data_test = objectnet.Objectnet(subset='not_imagenet',
                                        data_path=path_data_objectnet,
                                        model=model_class)
        data_valid = imagenet.Imagenet(data_path=path_data_imagenet,
                                       model=model_class)
    elif data_name == "Objectnet_only_imagenet":
        data_test = objectnet.Objectnet(subset='only_imagenet',
                                        data_path=path_data_objectnet,
                                        model=model_class)
        data_valid = imagenet.Imagenet(data_path=path_data_imagenet,
                                       model=model_class)
    # Load Model
    if data_name in ["Imagenet", "CIFAR"]:
        load_logits = True
    elif data_name in ["Objectnet_not_imagenet", "Objectnet_only_imagenet"]:
        load_logits = False
    modelf = ModelFactory(model_class, save_path_general=model_path,
                          data_corrupted_path=path_data_imagenet_corrupted)
    modelf.load(bool_load=model_load, load_path=model_path, load_logits=load_logits)
    modelf.save()

    return modelf, data_valid, data_test


def ifValueNotNone(default, value):
    """Helper function for argument parser"""
    if value is not None:
        return value
    else:
        return default


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("-settings", nargs="?", help="settings e.g. IMAGENET_ResNet50_OOD_GN")
    parser.add_argument("-calib_models", nargs="?", help="calib_models e.g. TSModel,ETSModel")
    parser.add_argument("--folder_path_save", nargs="?",
                        help="path where the model is stored e.g. ../results/Calibration_ood")
    parser.add_argument("--perturb_levels", type=int, nargs="?",
                        help="number of perturbation levels (incl. level where epsilon=0)")
    parser.add_argument("--path_data_imagenet", nargs="?", help="path to ImageNet dataset")
    parser.add_argument("--path_data_objectnet", nargs="?", help="path to ObjectNet dataset")
    parser.add_argument("--path_data_imagenet_corrupted", nargs="?", help="path to corrupted ImageNet dataset")

    args = parser.parse_args()
    settings = args.settings
    folder_path_save = ifValueNotNone("./", args.folder_path_save)
    models_cms = ifValueNotNone(None, [str(i) for i in args.calib_models.split(",")] if args.calib_models else None)
    perturb_levels = ifValueNotNone(10, args.perturb_levels)
    path_data_imagenet = ifValueNotNone("/", args.path_data_imagenet)
    path_data_objectnet = ifValueNotNone("/", args.path_data_objectnet)
    path_data_imagenet_corrupted = ifValueNotNone("/", args.path_data_imagenet_corrupted)

    # Models that are used
    models_cms_obj_list = []
    if models_cms is not None:
        if "BaseModel" in models_cms: models_cms_obj_list.append(BaseModel())
        if "TSModel" in models_cms: models_cms_obj_list.append(TSModel())
        if "TSModel" in models_cms: models_cms_obj_list.append(TSModel())
        if "ETSModel" in models_cms: models_cms_obj_list.append(ETSModel())
        if "IRMModel" in models_cms: models_cms_obj_list.append(IRMModel())
        if "IROVAModel" in models_cms: models_cms_obj_list.append(IROVAModel())
        if "IROVATSModel" in models_cms: models_cms_obj_list.append(IROVATSModel())

    print(settings)
    print(models_cms)
    print(models_cms_obj_list)

    perturbations = [
        "rot_left",
        "rot_right",
        "xshift",
        "yshift",
        "xyshift",
        "shear",
        "xzoom",
        "yzoom",
        "xyzoom"
    ]

    perturbations_imagenet = [
        "imagenet2012_corrupted/shot_noise",
        "imagenet2012_corrupted/impulse_noise",
        "imagenet2012_corrupted/defocus_blur",
        "imagenet2012_corrupted/glass_blur",
        "imagenet2012_corrupted/motion_blur",
        "imagenet2012_corrupted/zoom_blur",
        "imagenet2012_corrupted/snow",
        "imagenet2012_corrupted/frost",
        "imagenet2012_corrupted/fog",
        "imagenet2012_corrupted/brightness",
        "imagenet2012_corrupted/contrast",
        "imagenet2012_corrupted/elastic_transform",
        "imagenet2012_corrupted/pixelate",
        "imagenet2012_corrupted/jpeg_compression",
        "imagenet2012_corrupted/gaussian_blur",
        "imagenet2012_corrupted/saturate",
        "imagenet2012_corrupted/spatter",
        "imagenet2012_corrupted/speckle_noise",
    ]

    # ADJUST THIS FOR YOUR EXPERIMENTS #

    # list of experiment settings (dictionaries):
    experiment_settings = {

        # EXAMPLES:

        # Imagenet ResNet50
        ##In Domain Training
        "IMAGENET_ResNet50_InD": {
            "model_class": "ResNet50",
            "model_path": folder_path_save,  # path to classifier
            "model_load": False,  # whether the post-hoc tuning model should be loaded
            "data_name": "Imagenet",
            "cms": [BaseModel(), TSModel(), TSModel(), ETSModel(), IRMModel(),
                    IROVAModel(), IROVATSModel()],
            "tuning_perturb": False,
            "perturb_levels": perturb_levels,  # number of pertrubation levels (epsilons)
            "perturbations": perturbations_imagenet,
            "test_name": "test_ds",
            "folder_path_save": ifValueNotNone("../results/Calibration_indomain", folder_path_save),
            "path_data_imagenet": path_data_imagenet,
            "path_data_objectnet": path_data_objectnet,
            "path_data_imagenet_corrupted": path_data_imagenet_corrupted
        },

        ###Gaussian Noise Training
        "IMAGENET_ResNet50_OOD_GN": {
            "model_class": "ResNet50",
            "model_path": folder_path_save,  # path to classifier
            "model_load": False,  # whether the post-hoc tuning model should be loaded
            "data_name": "Imagenet",
            "cms": [BaseModel(), TSModel(), TSModel(), ETSModel(), IRMModel(),
                    IROVAModel(), IROVATSModel()],
            "tuning_perturb": True,
            "perturbations": perturbations_imagenet,
            "perturb_levels": perturb_levels,  # number of pertrubation levels (epsilons)
            "test_name": "test_ds",
            "folder_path_save": ifValueNotNone("../results/Calibration_gaussian", folder_path_save),
            "path_data_imagenet": path_data_imagenet,
            "path_data_objectnet": path_data_objectnet,
            "path_data_imagenet_corrupted": path_data_imagenet_corrupted
        },
    }

    i = 0
    n = len(experiment_settings.keys())
    # run experiments with nice console prints
    print("--------------------------")
    print("INITIALIZATION SUCCESSFUL.")
    print("--------------------------")
    # print("Exp-Settings", experiment_settings.items())
    # print("Chosen Settings: ", settings)
    chosen_settings = experiment_settings[settings]
    # noinspection PySimplifyBooleanCheck
    if models_cms_obj_list != []:
        chosen_settings["cms"] = models_cms_obj_list
    # print('models_cms_obj_list: ', models_cms_obj_list)
    # print(chosen_settings)
    print("Chosen Settings: ", chosen_settings)
    run_experiment(chosen_settings)

    print("--------------------------")
    print("All experiments finished.")
