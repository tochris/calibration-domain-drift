import os
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from .postevaluation.evaluator import Evaluator


class PostCalibrator:
    """
    Class responsible for post-hoc calibration (tuning and evaluating) for a
    given perturbation. This acts as the inducer in our framework.
    Different perturbations can be specified for evaluating, but only one
    is used for tuning.
    """
    # (ML theory recap: Inducer coupled with an optimizer and a loss function
    # takes a dataset and outputs a point in the parameter space (= model))
    # A loss and a hypothesis space have to be given by argument.
    #
    # Several perturbations should be given, but only one is used for tuning.
    # The loss function has to be given during the initialization and not at the
    # `tune` method due to two reasons:
    #     - if the user wants to switch the loss function after having already
    #         tuned, it does not make sense to resume tuning on the result
    #         of the first tuning run. So, the user has to init a new instance
    #         anyways.
    #     - it would not be possible to run the save method without having
    #         executed `tune` before. This is minor, but the major part is,
    #         that the `load` method does also not work without a given loss
    #         function, which does not make sense from the user perspective.
    #         (the user wants to load an old instance by only specifying "path")


    def __init__(
        self,
        calib_model,
        #loss_func,
        model_factory,
        perturb_generator,
        data_name="Data",
        folder_path="results/Calibration"
    ):
        """
        Initialize new evaluator instance and define folder path.
        Set calibration calib_model class, perturb_generator, data,
        model_factory.
        :param calib_model: CalibModel object, used for tuning the post-hoc
            calibrator and stores the tuned parameters (theta).
        :param data: tf.data.Dataset object (x_data, y_labels), prepared with
            batch_size, shuffle etc.
        :param perturb_generator: PerturbGenerator object, generates perturbations.
        :param folder_path: string, folder where the folder system should be
            based. It's recommended to specify it here instead of at the `save`
            method.
        """
        self.calib_model = calib_model
        #self.loss_func = loss_func
        self.evaluator = Evaluator()

        self.model_factory = model_factory
        self.perturb_generator = perturb_generator
        print("self.model_factory.save_path", self.model_factory.save_path)
        print(os.path.split(self.model_factory.save_path)[0])
        print(os.path.split(os.path.split(self.model_factory.save_path)[0])[1])
        print("SAVE_PATH: ", os.path.split(self.model_factory.save_path)[1])
        self.folder_path = os.path.join(
            folder_path,
            data_name,
            # extract model name with parameters
            os.path.split(os.path.split(self.model_factory.save_path)[0])[1],
            self.calib_model.folder,
            #self.loss_func.__name__
        )

    def tune(
        self,
        data,
        tuning_perturb,
        epsilons,
        from_cache=False,
        to_cache=False,
        save_to_file=False
    ):
        """
        Method to tune a respective post-hoc calibration model.
        :param data: tf.data.Dataset object (x_data, y_labels), prepared with
            batch_size, shuffle etc.
        :param tuning_perturb: bool, if True gaussian perturbation approach is used
        :param epsilons: [int], level of perturbation (higher = more extreme)
        :from_cache: bool, whether logits are read from cache
        :from_cache: bool, whether logits are written to cache
        :from_cache: bool, whether logits are stored in a file
        """

        if tuning_perturb == True:
            perturb_type = "general_gaussian_noise"
            epsilons = epsilons
        else:
            perturb_type = "None"
            epsilons = [0]

        # list of logit-label pairs over all epsilons
        logits_labels_eps = [
            self.model_factory.logits(
                data,
                self.perturb_generator,
                perturb_type,
                epsilon,
                from_cache=from_cache,
                to_cache=to_cache,
                save_to_file=save_to_file
            ) for epsilon in epsilons
        ]

        # sum of loss functions of theta across different epsilons
        # theta |-> sum([ loss( f(x_eps, theta), y_eps) ])
        # all our tuneable losses return list of length 1, so we index
        # that value

        # if self.calib_model.folder in ["TS",
        #     "ETS",
        #     "IRM",
        #     "IROVA",
        #     "IROVATS",
        #     "PBMC",]:

        logits = []
        labels = []
        for logits_e, labels_e in logits_labels_eps:
            logits.append(logits_e)
            labels.append(labels_e)
        logits = np.concatenate(logits)
        labels = np.concatenate(labels)
        #print('logits: ', np.shape(logits))
        #print('labels: ', np.shape(labels))
        result = self.calib_model.optimize(logits,labels)
        self.tuning_result = result

        # else:
        #     loss_theta = lambda theta: \
        #         np.mean([
        #             self.loss_func(
        #                 self.calib_model.function(logits, theta),
        #                 labels,
        #             )[0] for logits, labels in logits_labels_eps
        #         ])
        #     result = minimize(
        #         loss_theta,
        #         self.calib_model.theta,
        #         method="nelder-mead",
        #         options={"maxfev": maxfev},
        #     )
        #     # update the theta parameter with the tuning result
        #     self.calib_model.setTheta(result["x"])
        #     # store result for possible manual checking
        #     self.tuning_result = result
        print("Tuned Parameters: ", self.calib_model.theta)

    def evaluate(
        self,
        data,
        test_metric,
        perturb_type,
        epsilon,
        from_cache=True,
        to_cache=True,
        save_to_file=False,
        ):
        """
        Method to evaluate the tuned calibration calib_model on the outer measures.
        All outer measures are the same base function, but with different
        logits based on all available perturbations.
        :param data: tf.data.Dataset object (x_data, y_labels), prepared with
            batch_size, shuffle etc.
        :param test_metric: string, a metric such as 'brier_score'
        :param perturb_type: string, type of data perturbation
        :param epsilon: int, level of perturbation (higher = more extreme)
        :from_cache: bool, whether logits are read from cache
        :from_cache: bool, whether logits are written to cache
        :from_cache: bool, whether logits are stored in a file
        """
        logits, labels = self.model_factory.logits(
            data,
            self.perturb_generator,
            perturb_type,
            epsilon,
            from_cache=from_cache,
            to_cache=to_cache,
            save_to_file=save_to_file
        )

        # loss( f(x, theta), truth)
        eval_values = test_metric(
            self.calib_model.function(logits, self.calib_model.theta),
            labels
        )
        # extend evaluator storage with
        # {perturb_type: {epsilon: {test_metric_name: test_metric_values}}}
        perturb_type = perturb_type.replace('/', '_')
        self.evaluator.add(
            test_metric.__name__,
            perturb_type,
            epsilon,
            eval_values
        )
        #print(test_values)

    def predict(self,
                data,
                perturb_type,
                epsilon):
        """
        Method to calculate the predictions of the underlying model, given
        the data, perturbation type and perturbation level (epsilon)
        :param data: tf.data.Dataset object (x_data, y_labels), prepared with
            batch_size, shuffle etc.
        :param perturb_type: string, type of data perturbation
        :param epsilon: int, level of perturbation (higher = more extreme)
        :return predictions: np.array corresponding to data
        :return labels: np.array corresponding to data
        """
        logits, labels = self.model_factory.logits(
            data,
            self.perturb_generator,
            perturb_type,
            epsilon,
            from_cache=False,
            to_cache=False,
            save_to_file=False
        )
        logits = self.calib_model.function(logits, self.calib_model.theta)
        preds = softmax(logits, axis=1)
        return preds, labels

    def save(self, folder_path=None):
        """
        Method to save the evaluation storage object.
        :param folder_path: string, should rather not be defined here, but instead
            at the class instance initialization. The file path will be appended by
            two levels: calib model and loss func.
        """
        if folder_path is None:
            folder_path = self.folder_path
        self.evaluator.save(folder_path)

    def reset(self):
        """
        Deletes the old evaluator object (and its storage) and replaces it
        with a newly initialized one.
        Also deletes logit and label dictionaries in the model factory.
        Careful: No save or backup is happening, so make sure nothing is lost.
        """
        self.evaluator = Evaluator()
        self.model_factory.logits_test = {}
        self.model_factory.labels_test = {}

    # def load(self, folder_path=None):
    #     """
    #     Method to load the calib model parameters and the evaluation storage
    #     object from a file path.
    #     The file path will be appended by two levels: calib model and loss func
    #     :param folder_path: string, should not be defined here, but instead
    #     at the class instance initialization.
    #     """
    #     if folder_path is None:
    #         folder_path = self.folder_path
    #
    #     self.calib_model.load(folder_path)
    #     self.evaluator.load(folder_path)

    # def generate_plots(self, folder_path=None, **kwargs):
    #     """
    #     Method to generate several plots at once.
    #     The plots will be returned, but also stored to a folder system under
    #     the given folder path.
    #     Currently only line plots will be generated, but this can easily
    #     be extended in the Evaluator class.
    #     :param folder_path: string, should not be defined here, but instead
    #     at the class instance initialization.
    #     :return: Nested dictionary of seaborn plot objects
    #     """
    #     if folder_path is None:
    #         folder_path = self.folder_path
    #
    #     plots = {
    #         "lineplots": self.evaluator.generate_lineplots(
    #             path=folder_path, **kwargs
    #         ),
    #         "boxplots": self.evaluator.generate_boxplots(
    #             path=folder_path, loss_type="confidence_scores", **kwargs
    #         )
    #     }
    #     return plots
