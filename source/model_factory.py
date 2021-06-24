import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append("./")

from .utils.log_functions import Log_extend_array
from .utils.plotutils import (
    save_dict_to_pkl,
    load_dict_from_pkl,
)


class ModelFactory:
    """
    A template for pre-defined classifier models.
    Construct classifier model object via .load(...)
    """

    def __init__(
            self,
            architecture,
            save_path_general=False,
            data_corrupted_path=None,
            **kwargs
    ):
        """
        Define architecture and folder path.
        :param architecture: string, defines the model's architecture:
                Options: e.g. 'ResNet50'
        :param save_path_general: string, path to save the trained models
        :param data_corrupted_path: string, path to corrupted data
        """
        self.architecture = architecture

        if save_path_general is not None:
            self.save_path = os.path.join(save_path_general, architecture + "/")
            self.log_dir_test = self.save_path
            directory = os.path.dirname(self.save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            print("SAVE-PATH: ", self.save_path)

        self.data_corrupted_path = data_corrupted_path

        self.logits_test = {}
        self.labels_test = {}

    def load(self,
             load_path=None,
             bool_load=False,
             compile=True,
             load_logits=False,
             **kwargs):
        """
        Loads the model either from load_path or from tf.keras
        (if bool_load==False)
        load_path: string, determines the path to the model's folder
        compile: bool, if True tf.keras.model gets compiled
        load_logits: bool, if True logits are loaded from file, otherwise
            they are computed
        """

        if not bool_load:
            if self.architecture == "ResNet50":
                self.model = tf.keras.applications.ResNet50(
                    weights="imagenet",
                    classes=1000
                )
            elif self.architecture == "ResNet152":
                self.model = tf.keras.applications.ResNet152(
                    weights="imagenet",
                    classes=1000
                )
            elif self.architecture == "VGG19":
                self.model = tf.keras.applications.VGG19(
                    weights="imagenet",
                    classes=1000
                )
            elif self.architecture == "DenseNet169":
                self.model = tf.keras.applications.DenseNet169(
                    weights="imagenet",
                    classes=1000
                )
            elif self.architecture == "EfficientNetB7":
                self.model = tf.keras.applications.EfficientNetB7(
                    weights="imagenet",
                    classes=1000
                )
            elif self.architecture == "Xception":
                self.model = tf.keras.applications.Xception(
                    weights="imagenet",
                    classes=1000
                )
            elif self.architecture == "MobileNetV2":
                self.model = tf.keras.applications.MobileNetV2(
                    weights="imagenet",
                    classes=1000
                )
        else:
            assert load_path is not None, "load path not specified!"
            path = os.path.join(load_path, "model.h5")
            self.load_model(path)
            self.save_path = load_path
            print("SAVE-PATH: ", self.save_path)

        if compile:
            self.model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()],
            )

        if load_logits:
            if os.path.isfile(
                    self.log_dir_test + "logits_test" + ".pkl"
            ) and os.path.isfile(self.log_dir_test + "labels_test" + ".pkl"):
                self.logits_test = load_dict_from_pkl(self.log_dir_test, "logits_test")
                self.labels_test = load_dict_from_pkl(self.log_dir_test, "labels_test")

    def test_step_logits(self, x_data, labels, log_dict):
        def tf_test_step_logits():
            preds = self.model.predict(x_data)
            logits = tf.math.log(preds)
            log_dict["accuracy"](labels, preds)
            log_dict["logits"].add(logits)
            log_dict["labels"].add(labels)

        tf_test_step_logits()
        return log_dict

    # noinspection PyDictCreation
    def test(
            self,
            dataset,
            perturbation_generator=None,
            perturb_type=None,
            epsilon=None,
            cache_results=False,
    ):
        """
        test model, optionally based on perturbation;
        :param dataset: tf.data.Dataset object of (x_data, y_labels), prepared with
        batch_size etc.
        :param perturbation_generator: PerturbGenerator object, generates
            perturbations.
        :param perturb_type:  string, type of data perturbation
        :param epsilon: int, level of perturbation (higher = more extreme)
        :param cache_results: bool, if True logits and labels are stored temporary
        :return logits: np.array corresponding to dataset
        :return labels: np.array corresponding to dataset
        """

        # create log metrics
        log_dir_test_detailed = (
                self.log_dir_test + str(perturb_type) + "_" + str(epsilon) + "/"
        )
        log_dict_test = {}
        log_dict_test["accuracy"] = tf.keras.metrics.CategoricalAccuracy(
            name="accuracy"
        )
        log_dict_test["logits"] = Log_extend_array(name="logits")
        log_dict_test["labels"] = Log_extend_array(name="labels")

        # get corrupted tf.data if applicable
        batch_perturb = True
        if perturbation_generator is not None:
            dataset_perturb = perturbation_generator.perturb_dataset(
                dataset=dataset,
                perturb_type=perturb_type,
                epsilon=epsilon,
                data_corrupted_path=self.data_corrupted_path,
                model=self.architecture
            )
            if dataset_perturb is None:
                batch_perturb = True
            else:
                batch_perturb = False
                dataset = dataset_perturb

        if not batch_perturb:
            preds = self.model.predict(dataset)
            logits = np.log(preds)
            labels = np.concatenate([y for x, y in dataset], axis=0)

            log_dict_test["accuracy"](labels, preds)
            log_dict_test["logits"].add(logits)
            log_dict_test["labels"].add(labels)

        else:
            for x_data, y_data in dataset:
                if batch_perturb:
                    if perturbation_generator is not None:
                        x_data = perturbation_generator.perturb_batch(
                            x_data,
                            perturb_type=perturb_type,
                            epsilon=epsilon,
                            model=self
                        )
                log_dict_test = self.test_step_logits(x_data, y_data, log_dict_test)

        logits_test = log_dict_test["logits"].result()
        labels_test = log_dict_test["labels"].result()
        if cache_results:
            self.logits_test[str(perturb_type) + "_" + str(epsilon)] = logits_test
            self.labels_test[str(perturb_type) + "_" + str(epsilon)] = labels_test

        # plot results
        keys_to_plot = {"accuracy"}
        log_dict_test_plot = {
            key + "_train": value.result().numpy()
            for key, value in log_dict_test.items()
            if key in keys_to_plot
        }
        # print several metrics per epoch
        print(
            "Test Accuracy= "
            + "{:.3f}".format(log_dict_test["accuracy"].result().numpy())
        )

        return logits_test, labels_test

    def logits(
            self,
            dataset=None,
            perturb_generator=None,
            perturb_type=None,
            epsilon=None,
            from_cache=False,
            to_cache=False,
            save_to_file=False,
    ):
        """
        dataset: tf.data.Dataset object of (x_data, y_labels), prepared with
        batch_size etc.
        :param perturbation_generator: PerturbGenerator object, generates
            perturbations.
        :param perturb_type:  string, type of data perturbation
        :param epsilon: int, level of perturbation (higher = more extreme)
        :from_cache: bool, whether logits are read from cache
        :from_cache: bool, whether logits are written to cache
        :from_cache: bool, whether logits are stored in a file
        :return logits: np.array corresponding to dataset
        :return labels: np.array corresponding to dataset
        """
        key = str(perturb_type) + "_" + str(epsilon)
        if from_cache:
            if key in self.logits_test.keys() and key in self.labels_test.keys():
                logits_test, labels_test = self.logits_test[key], self.labels_test[key]
            else:
                if dataset is None:
                    raise ValueError("Dataset is not specified!")
                logits_test, labels_test = self.test(
                    dataset, perturb_generator, perturb_type, epsilon, to_cache
                )
        else:
            if dataset is None:
                raise ValueError("Dataset is not specified!")
            logits_test, labels_test = self.test(
                dataset, perturb_generator, perturb_type, epsilon, to_cache
            )
        if save_to_file:
            save_dict_to_pkl(self.log_dir_test, self.logits_test, "logits_test")
            save_dict_to_pkl(
                self.log_dir_test, self.labels_test, "labels_test"
            )
        return logits_test, labels_test

    def save_logits(self):
        """
        Saves logits and labels to specified path.
        """
        print("save logits to: ", self.log_dir_test)
        save_dict_to_pkl(self.log_dir_test, self.logits_test, "logits_test")
        save_dict_to_pkl(
            self.log_dir_test, self.labels_test, "labels_test"
        )

    def save(self):
        """
        Saves classifier model to specified path.
        """
        print("save model...")
        self.model.save(os.path.join(self.save_path, "model.h5"))

    def load_model(self, path):
        """
        Loads classifier model from specified path.
        """
        print('load model from following path: ', path)
        self.model = tf.keras.models.load_model(path)
        print("Model was loaded from file.")
