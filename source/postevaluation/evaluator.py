import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import source.postevaluation.calibration as calib
from ..utils.helpers import load_obj, save_obj
from ..utils.calibration import *


class Evaluator:
    """
    Class for test result storage and further methods for evaluation of that.
    Storage is a pandas dataframe of test metric results.
    """

    def __init__(self):
        self.storage = pd.DataFrame(
            columns = ["Loss_type", "Index", "Perturbation", "Epsilon", "Value"]
        )


    def add(self, loss_type, perturbation, epsilon, values):
        """
        Appends the storage data frame with a new row.
        :param loss_type: String, which loss is the value based on?
        :param perturbation: String, which perturbation type was used
        :param epsilon: Index, the perturbation level
        :param values: [Float], values of the loss function; an additional
                        index is stored to specify the position of each
                        value in the list
        """
        n = len(values)
        to_append = pd.DataFrame({
            "Loss_type": np.repeat(loss_type, n),
            "Index": range(n),
            "Perturbation": np.repeat(perturbation, n),
            "Epsilon": np.repeat(epsilon, n),
            "Value": values
        })

        self.storage = pd.concat(
            [self.storage, to_append],
            ignore_index = True
        )

    def save(self, path = "./", file = "evaluator_storage.pkl", with_csv=False):
        """
        Method to save the evaluation storage object.
        :param path: string, directory path of where the target is located
        :param file: string, target file name
        :param with_csv: bool, also save the storage as a csv file
        """
        if not os.path.exists(path):
            os.makedirs(path)

        if with_csv:
            # safe each loss type as csv for additional readability
            for lt in set(self.storage.Loss_type):
                df = self.storage
                df.loc[df.Loss_type==lt].to_csv(path + lt + ".csv")

        save_obj(self.storage, path, file)

    def load(self, path = "./", file = "evaluator_storage.pkl"):
        """
        Method to load the evaluation storage object from a file.
        :param path: string, directory path of where the target is located
        :param file: string, target file name
        """
        self.storage = load_obj(path, file)

    def remove_duplicates(self):
        """
        remove evaluations done more than once (the Value is not considered)
        the latest evaluation is kept
        """
        self.storage.drop_duplicates(
            ["Loss_type", "Index", "Perturbation", "Epsilon"],
            keep="last",
            inplace=True
        )

    def mean(self, Loss_type=None, Perturbation=None, Epsilon=None, Index=None):
        df = self.storage
        if Loss_type is not None:
            df = df.loc[df.Loss_type==Loss_type]

        if Perturbation is not None:
            df = df.loc[df.Perturbation==Perturbation]

        if Epsilon is not None:
            df = df.loc[df.Epsilon==Epsilon]

        if Index is not None:
            df = df.loc[df.Index==Index]

        return df.Value.mean()

    def micro_avg_ece(
        self,
        Perturbation=None,
        bins_limits=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ):

        df = self.storage
        assert "confidence_scores" in set(df.Loss_type)
        assert "matches" in set(df.Loss_type)

        if Perturbation is not None:
            df = df.loc[df.Perturbation==Perturbation]

        binned_prob, binned_acc = calib.calculate_binned_prob_acc_list(
            df[df.Loss_type=="confidence_scores"].Value,
            df[df.Loss_type=="matches"].Value,
            bins_limits
        )

        return calib.calculate_expected_calibration_error(
            binned_prob,
            binned_acc,
            df[df.Loss_type=="confidence_scores"].Value,
            bins_limits
        )

    def generate_lineplots(
        self, perturbation=None, loss_type=None, path="./"
    ):
        """
        Creates a nested dictionary of seaborn plot objects
        based on the storage data frame.
        The figures are additionally stored in a folder system.
        :param x_axis: String, column name used for the x axis
        :param y_axis: String, column name used for the y axis
        :param perturbation: String, name of the perturbation to select on
        :return: Nested dictionary of seaborn plot objects,
        first level: perturbation types, second level: loss types
        """
        assert self.storage.size > 0, "Cannot plot an empty storage."

        # if no arguments are given, plot everything in storage
        if perturbation is None:
            perturbations = set(self.storage["Perturbation"])
        else:
            perturbations = [perturbation]

        if loss_type is None:
            loss_types = set(self.storage["Loss_type"])
        else:
            loss_types = [loss_type]

        # init return object
        plot_dict = {}

        for loss in loss_types:
            plot_dict[loss] = {}

            # calculating the plot limits
            loss_data = self.storage.loc[
                (self.storage["Loss_type"] == loss), :
            ]
            loss_min = loss_data.min()["Value"]
            loss_max = loss_data.max()["Value"]
            border = 0.05 * (loss_max - loss_min)

            for perturb in perturbations:
                # define subset of storage data for plotting
                plot_data = loss_data.loc[
                    (self.storage["Perturbation"] == perturb), :
                ]
                plot = sb.lineplot(
                    x = plot_data["Epsilon"], y = plot_data["Value"]
                )
                plot.set(ylim = (
                     loss_min - border,
                     loss_max + border
                ))
                plot_dict[loss][perturb] = plot
                # save plot to foldersystem
                f_path = os.path.join(path, loss + "_eval", "lineplot")
                if not os.path.exists(f_path):
                    os.makedirs(f_path)

                figure = plot.get_figure()
                figure.savefig(
                    os.path.join(f_path, perturb + ".svg")
                )
                figure.clf()

        return plot_dict


    def reliability_diagram(
        self, perturbation=None, epsilon=None, bins_calibration=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ):
        """
        storage - evaluator storage object
        """
        storage = self.storage

        if perturbation is not None:
            storage = storage.loc[
                (storage["Perturbation"] == perturbation), :
            ]

        if epsilon is not None:
            storage = storage.loc[
                (storage["Epsilon"] == epsilon), :
            ]

        conf_score_df = storage.loc[
            (storage["Loss_type"] == "confidence_scores"), :
        ]
        matches_df = storage.loc[
            (storage["Loss_type"] == "matches"), :
        ]
        probability_vec_test = conf_score_df["Value"]
        match_vec_test = matches_df["Value"]

        prob_binned, accs_binned = \
            calib.calculate_binned_prob_acc_list(
                probability_vec_test,
                match_vec_test, bins_calibration
            )

        calibration_bar_chart(prob_binned, accs_binned[::-1])


    def generate_boxplots(
        self, perturbation=None, loss_type=None, path="./"
    ):
        """
        Creates a nested dictionary of seaborn plot objects
        based on the storage data frame.
        The figures are additionally stored in a folder system.
        :param x_axis: String, column name used for the x axis
        :param y_axis: String, column name used for the y axis
        :param perturbation: String, name of the perturbation to select on
        :return: Nested dictionary of seaborn plot objects,
        first level: perturbation types, second level: loss types
        """
        assert self.storage.size > 0, "Cannot plot an empty storage."

        # if no arguments are given, plot everything in storage
        if perturbation is None:
            perturbations = set(self.storage["Perturbation"])
        else:
            perturbations = [perturbation]

        if loss_type is None:
            loss_types = set(self.storage["Loss_type"])
        else:
            loss_types = [loss_type]

        # init return object
        plot_dict = {}

        for loss in loss_types:
            plot_dict[loss] = {}

            # calculating the plot limits
            loss_data = self.storage.loc[
                (self.storage["Loss_type"] == loss), :
            ]
            loss_min = loss_data.min()["Value"]
            loss_max = loss_data.max()["Value"]
            border = 0.05 * (loss_max - loss_min)

            for perturb in perturbations:
                # define subset of storage data for plotting
                plot_data = loss_data.loc[
                    (self.storage["Perturbation"] == perturb), :
                ]
                plot = sb.boxplot(
                    x = plot_data["Epsilon"], y = plot_data["Value"]
                )
                plot.set(ylim = (
                    loss_min - border,
                    loss_max + border
                ))
                plot_dict[loss][perturb] = plot
                # save plot to foldersystem
                f_path = os.path.join(path, loss + "_eval", "boxplot")
                if not os.path.exists(f_path):
                    os.makedirs(f_path)

                figure = plot.get_figure()
                figure.savefig(
                    os.path.join(f_path, perturb + ".svg")
                )
                figure.clf()

        return plot_dict

    def class_prob_plot():
        class_probabilities(
            class_prob_vec_step,
        )

    def freq_plot(
        self, perturbation=None, epsilon=None, bins_calibration=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    ):
        storage = self.storage

        if perturbation is not None:
            storage = storage.loc[
                (storage["Perturbation"] == perturbation), :
            ]

        if epsilon is not None:
            storage = storage.loc[
                (storage["Epsilon"] == epsilon), :
            ]

        conf_score_df = storage.loc[
            (storage["Loss_type"] == "confidence_scores"), :
        ]

        frequency_over_calibration(
            conf_score_df["Value"],
            bins_calibration
        )

    def ece_curve(self, perturbation=None):
        """
        returns seaborn plot object
        """
        storage = self.storage

        if perturbation is not None:
            storage = storage.loc[
                (storage["Perturbation"] == perturbation), :
            ]

        eces = storage.loc[
            (storage.Loss_type == "ECE"),
            ["Value", "Epsilon"]
        ]
        eces = eces.groupby("Epsilon").mean()
        sb_plot = sb.lineplot(x=range(10), y=eces.Value)

        return sb_plot

    def acc_curve(self, perturbation=None):
        """
        returns seaborn plot object
        """
        storage = self.storage

        if perturbation is not None:
            storage = storage.loc[
                (storage["Perturbation"] == perturbation), :
            ]

        eces = storage.loc[
            (storage.Loss_type == "accuracy"),
            ["Value", "Epsilon"]
        ]
        eces = eces.groupby("Epsilon").mean()
        sb_plot = sb.lineplot(x=range(10), y=eces.Value)

        return sb_plot

    def boxplot(self, Loss_type, perturbation=None):
        """
        Boxplots of a chosen 'Loss_type' for each epsilon.
        May restrict to only use results of a specific perturbation.
        return seaborn plot object
        """

        storage = self.storage

        if perturbation is not None:
            storage = storage.loc[
                (storage["Perturbation"] == perturbation), :
            ]

        storage = storage.loc[
            (storage["Loss_type"] == Loss_type), :
        ]

        sb_plot = sb.boxplot(
            x=storage["Epsilon"], y=storage["Value"]
        )

        return sb_plot


    def lineplot(
        self, loss_type_x, loss_type_y, perturbation=None, epsilon=None
    ):
        """
        Creates a lineplot with error bars.
        loss_type_x is only allowed to have Index 0.
        loss_type_y can have higher Index values - the mean and sd statistics
        are then calculated based on these indeces.
        return seaborn plot object
        """

        storage = self.storage

        if perturbation is not None:
            storage = storage.loc[
                (storage["Perturbation"] == perturbation), :
            ]

        if epsilon is not None:
            storage = storage.loc[
                (storage["Epsilon"] == epsilon), :
            ]

        xs = storage.loc[storage["Loss_type"] == loss_type_x, :].Value
        ys = storage.loc[storage["Loss_type"] == loss_type_y, :]
        # replicate values of xs to make lists equal in length
        xs = [x for x in xs for _ in set(ys.Index)]

        plot_data = pd.DataFrame({
            "x": xs,
            "y": ys.Value
        })

        return sb.lineplot(x="x", y="y", data=plot_data)


    def binned_plot(self, binned_loss, perturbation=None, epsilon=None):

        storage = self.storage

        if perturbation is not None:
            storage = storage.loc[
                (storage["Perturbation"] == perturbation), :
            ]

        if epsilon is not None:
            storage = storage.loc[
                (storage["Epsilon"] == epsilon), :
            ]

        subset = storage.loc[
            (storage.Loss_type == binned_loss),
            ["Value", "Index"]
        ]

        subset = subset.astype("float")
        subset = subset.groupby("Index").mean()
        df = pd.DataFrame({"bin": range(10), "loss": subset.Value})

        return sb.lineplot(x="bin", y="loss", data=df)

    def temperature_plot(self, calib_func, perturbation=None):

        storage = self.storage

        if perturbation is not None:
            storage = storage.loc[
                (storage["Perturbation"] == perturbation), :
            ]

        subset = storage.loc[
            (storage.Loss_type == "ranges"),
            ["Value", "Index", "Epsilon"]
        ]

        logits = [[0, v] for v in subset.Value]
        calibs = calib_func(logits)
        temps = [np.divide(l[1], c[1]) for l, c in zip(logits, calibs)]

        # get temperature values as y-axis
        plot_data = pd.DataFrame({
            "x": subset.Epsilon,
            "y": temps
        })

        return sb.boxplot(x="x", y="y", data=plot_data)

    def dist_plot(self, loss_type, perturbation=None, epsilon=None, **kwargs):

        storage = self.storage

        if perturbation is not None:
            storage = storage.loc[
                (storage["Perturbation"] == perturbation), :
            ]

        if epsilon is not None:
            storage = storage.loc[
                (storage["Epsilon"] == epsilon), :
            ]

        subset = storage.loc[
            (storage.Loss_type == loss_type),
            ["Value"]
        ]

        return sb.distplot(subset.Value, **kwargs)
