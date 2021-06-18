import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy.random import default_rng
import source.data.imagenet_corrupted as imagenet_corrupted
from tensorflow.python.ops import array_ops


class PerturbationGenerator:
    """Class that handels perturbation of data."""

    def __init__(self, dataset_name='Imagenet', gaussian_noise_eps=None, **kwargs):
        """
        Set perturbation parameters
        Args:
            dataset_name: string, e.g. CIFAR10, Imagenet
            gaussian_noise: list [float], optimized levels of perturbation
                (=epsilons) for gaussian perturbation tuning approach.
        """

        self.dataset_name = dataset_name
        self.image_generator = ImageDataGenerator()
        # set params
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        # Parameters for gaussian perturbation tuning approach
        if gaussian_noise_eps is not None:
            self.params_perturb["general_gaussian_noise"] = np.array(gaussian_noise_eps)

        # Perturbations
        self.params_perturb = {}
        # Rotation angle in degrees (right)
        self.params_perturb["rot_right"] = [
            0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        # Rotation angle in degrees (left)
        self.params_perturb["rot_left"] = [
            0, 350, 340, 330, 320, 310, 300, 290, 280, 270]
        # Shift in the x direction
        self.params_perturb["xshift"] = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        # Shift in the y direction
        self.params_perturb["yshift"] = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        # Shift in the xy direction
        self.params_perturb["xyshift"] = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        # Shear angle in degrees
        self.params_perturb["shear"] = [
            0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        # Zoom in the x direction
        self.params_perturb["xzoom"] = [
            1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
        # Zoom in the y direction
        self.params_perturb["yzoom"] = [
            1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
        # Zoom in the xy direction
        self.params_perturb["xyzoom"] = [
            1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
        # Swap words randomly
        self.params_perturb["char_swap"] = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # map perturbation to keras fctn parameter
        self.params_perturb_keras = {}
        self.params_perturb_keras["rot_right"] = ["theta"]
        self.params_perturb_keras["rot_left"] = ["theta"]
        self.params_perturb_keras["shear"] = ["shear"]
        self.params_perturb_keras["xzoom"] = ["zx"]
        self.params_perturb_keras["yzoom"] = ["zy"]
        self.params_perturb_keras["xyzoom"] = ["zx", "zy"]
        self.params_perturb_keras["xshift"] = ["tx"]
        self.params_perturb_keras["yshift"] = ["ty"]
        self.params_perturb_keras["xyshift"] = ["tx", "ty"]

        self.perturb_corruption = ["imagenet2012_corrupted/gaussian_noise",
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
                                   "imagenet2012_corrupted/speckle_noise"]

        #self.param_ids = list(locals().keys())

    def possible_epsilons(self, perturb_type):
        """
        Return all possible epsilons
        """
        if perturb_type == "None":
            return [1]
        elif perturb_type in self.perturb_corruption:
            return [0, 1, 2, 3, 4, 5]
        else:
            return self.params_perturb[perturb_type]

    def perturb_dataset(self,
                        dataset,
                        perturb_type,
                        epsilon,
                        data_corrupted_path,
                        model=None):
        """
        Returns a perturbed tf.data object from corrupted dataset based on data_corrupted_path
        Args:
            dataset: tf.data.Dataset object (x_data, y_labels), prepared with
                batch_size, shuffle etc.
            perturb_type: string, type of data perturbation
            epsilon: int, level of perturbation
            data_corrupted_path: string, path to corrupted dataset
            model: tf.keras.model, classifier model
        Return:
            corrupted dataset: tf.data.Dataset object (x_data, y_labels), prepared
                with batch_size, shuffle etc.
        """
        if perturb_type in self.perturb_corruption:
            if epsilon != 0:
                data = imagenet_corrupted.Imagenet_corrupted(
                    corruption_type=perturb_type,
                    epsilon=epsilon,
                    data_path=data_corrupted_path,
                    model=model
                )
                return data.test_ds
            else:
                return dataset
        return None

    def perturb_batch(self,
                      X,
                      perturb_type=None,
                      epsilon=0,
                      model=None):
        """
        Returns perturbed data batch
        Args:
            X: array, input data batch
            dataset: tf.data.Dataset object (x_data, y_labels), prepared with
                batch_size, shuffle etc.
            perturb_type: string, type of data perturbation
            epsilon: int, level of perturbation
            model: tf.keras.model, classifier model
        Return:
            x_data: numpy array, perturbed data batch
        """
        # get batch of samples, return perturbed samples
        if perturb_type is None or perturb_type == "None":
            return X

        elif perturb_type in list(self.params_perturb_keras.keys()):
            Xnp = X.numpy()
            keras_paras = self.params_perturb_keras[perturb_type]
            keras_param_dict = {}
            for para in keras_paras:
                keras_param_dict[para] = self.params_perturb[perturb_type][epsilon]
            X_pert = []
            for x in Xnp:
                X_pert.append(
                    tf.keras.preprocessing.image.apply_affine_transform(
                        x, **keras_param_dict
                    )[np.newaxis, ...]
                )
            X_pert = np.concatenate(X_pert)

        elif perturb_type == "general_gaussian_noise":
            if self.dataset_name == 'Imagenet':
                mean = np.ones((array_ops.shape(X)[1], array_ops.shape(X)[2], 3), dtype=np.float32)
                mean[..., 0] = 123.68
                mean[..., 1] = 116.779
                mean[..., 2] = 103.939
                X = X[..., ::-1]
                X = X + mean
                X = X / 255.0
                eps = self.params_perturb[perturb_type][epsilon]
                eps = tf.random.normal(array_ops.shape(X), 0., eps)
                X_pert = X + eps
                X_pert = tf.clip_by_value(X_pert, 0, 1)
                X_pert = X_pert * 255.0
                X_pert = X_pert - mean
                X_pert = X_pert[..., ::-1]
            else:
                eps = self.params_perturb[perturb_type][epsilon]
                eps = tf.random.normal(array_ops.shape(X), 0., eps)
                eps = tf.cast(eps, dtype=tf.float64)
                X = tf.cast(X, dtype=tf.float64)
                X_pert = X + eps

        return X_pert

    # def adv_batch(self, X, perturb_type=None, epsilon=0, model=None, label=None):
    #     # get batch of samples, return adversatrial sample with desired epsilon
    #     if perturb_type is None or perturb_type == "None":
    #         return X
    #
    #     elif perturb_type == "fgsm":
    #         assert (
    #                 model is not None
    #         ), "Provide model from model factory to generate adversarials"
    #         assert label is not None, "Provide label to generate adversarials"
    #         signed_grad = self.get_signed_grad(X, label, model)
    #         X_pert = X + epsilon * signed_grad
    #
    #     return X_pert

    # def get_signed_grad(self, X, label, model):
    #     with tf.GradientTape() as tape:
    #         tape.watch(X)
    #         loss = model.loss_adv(X, label)
    #
    #     # Get the gradients of the loss w.r.t to the input image.
    #     gradient = tape.gradient(loss, X)
    #     # Get the sign of the gradients to create the perturbation
    #     signed_grad = tf.sign(gradient)
    #     return signed_grad
    #
    # def print_perturbations(self):
    #     # list all implemented pertubations
    #     print(self.params_perturb.keys())
