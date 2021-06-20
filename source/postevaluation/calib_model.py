import tensorflow as tf
import numpy as np
import os
from scipy.stats import entropy
from scipy.special import softmax
from netcal.scaling import LogisticCalibration
from netcal.binning import HistogramBinning
from ..postevaluation.utils_vuc_clibration import PlattBinnerMarginalCalibrator
from ..postevaluation.util_calibration_methods import \
    train_temperature_scaling, \
    train_ensemble_temperature_scaling, \
    train_isotonic_regression, \
    train_irova, \
    train_irovats, \
    calibrate_temperature_scaling, \
    calibrate_ensemble_temperature_scaling, \
    calibrate_isotonic_regression, \
    calibrate_irova, \
    calibrate_irovats


class BaseModel:
    """
    Base class for all post-hoc calibration models.
    """

    def __init__(self, init_theta=[0], folder="Base"):
        self.theta = init_theta
        self.folder = folder

    def function(self, logits, theta=None):
        """
        Calculate probabilities. For the base model this is just an
        identity function
        """
        if theta is None:
            theta = self.theta
        return logits


class TSModel(BaseModel):
    """
    Base class extended for Temperature Scaling.
    """

    def __init__(self, init_theta=None, folder="TS"):
        super().__init__(init_theta=init_theta, folder=folder)

    def function(self, logits, theta=None):
        """
        Calculate transformed probabilities.
        :param logits: tensor containing the logits of a single perturbation
        :param theta: model parameters
        :return: transformed logits
        """
        if theta is None:
            theta = self.theta
        calibrated_probs = calibrate_temperature_scaling(logits, theta)
        return np.log(np.clip(calibrated_probs, 1e-20, 1 - 1e-20))

    def optimize(self, logits, labels):
        """
        Tune post-hoc calibration model.
        :param logits: tensor containing the logits of a single perturbation
        :param labels: tensor containing the ground truth labels
        """
        self.theta = train_temperature_scaling(logits, labels, loss='ce')
        print(self.theta)


class ETSModel(BaseModel):
    """
    Base class extended for Ensemble Temperature Scaling.
    """

    def __init__(self, init_theta=None, folder="ETS"):
        super().__init__(init_theta=init_theta, folder=folder)

    def function(self, logits, theta=None):
        """
        Calculate transformed probabilities
        :param logits: tensor containing the logits of a single perturbation
        :param theta: model parameters
        :return: transformed logits
        """
        if theta is None:
            theta = self.theta
        t, w = self.theta
        calibrated_probs = calibrate_ensemble_temperature_scaling(logits, t, w, n_class=self.n_class)
        return np.log(np.clip(calibrated_probs, 1e-20, 1 - 1e-20))

    def optimize(self, logits, labels):
        """
        Tune post-hoc calibration model.
        :param logits: tensor containing the logits of a single perturbation
        :param labels: tensor containing the ground truth labels
        """
        labels_1d = np.argmax(labels, axis=1)
        self.n_class = np.max(labels_1d) + 1
        self.theta = train_ensemble_temperature_scaling(logits, labels, n_class=self.n_class, loss='mse')


class IRMModel(BaseModel):
    """
    Base class extended for multiclass isotonic regression.
    """

    def __init__(self, init_theta=None, folder="IRM"):
        super().__init__(init_theta=init_theta, folder=folder)

    def function(self, logits, theta=None):
        """
        Calculate transformed probabilities
        :param logits: tensor containing the logits of a single perturbation
        :param theta: model parameters
        :return: transformed logits
        """
        if theta is None:
            theta = self.theta
        calibrated_probs = calibrate_isotonic_regression(logits, theta)
        return np.log(np.clip(calibrated_probs, 1e-20, 1 - 1e-20))

    def optimize(self, logits, labels):
        """
        Tune post-hoc calibration model.
        :param logits: tensor containing the logits of a single perturbation
        :param labels: tensor containing the ground truth labels
        """
        self.theta = train_isotonic_regression(logits, labels)


class IROVAModel(BaseModel):
    """
    Base class extended for IRovA.
    """

    # we simply want a new default argument - otherwise this is not required
    def __init__(self, init_theta=None, folder="IROVA"):
        super().__init__(init_theta=init_theta, folder=folder)

    def function(self, logits, theta=None):
        """
        Calculate transformed probabilities
        :param logits: tensor containing the logits of a single perturbation
        :param theta: model parameters
        :return: transformed logits
        """
        if theta is None:
            theta = self.theta
        calibrated_probs = calibrate_irova(logits, theta)
        return np.log(np.clip(calibrated_probs, 1e-20, 1 - 1e-20))

    def optimize(self, logits, labels):
        """
        Tune post-hoc calibration model.
        :param logits: tensor containing the logits of a single perturbation
        :param labels: tensor containing the ground truth labels
        """
        self.theta = train_irova(logits, labels)


class IROVATSModel(BaseModel):
    """
    Base class extended for IRovA+TS.
    """

    # we simply want a new default argument - otherwise this is unrequired
    def __init__(self, init_theta=None, folder="IROVATS"):
        super().__init__(init_theta=init_theta, folder=folder)

    def function(self, logits, theta=None):
        """
        Calculate transformed probabilities
        :param logits: tensor containing the logits of a single perturbation
        :param theta: model parameters
        :return: transformed logits
        """
        if theta is None:
            theta = self.theta
        t, list_ir = theta
        calibrated_probs = calibrate_irovats(logits, t, list_ir)
        return np.log(np.clip(calibrated_probs, 1e-20, 1 - 1e-20))

    def optimize(self, logits, labels):
        """
        Tune post-hoc calibration model.
        :param logits: tensor containing the logits of a single perturbation
        :param labels: tensor containing the ground truth labels
        """
        self.theta = train_irovats(logits, labels, loss='mse')
