# Post-hoc Uncertainty Calibration for Domain Drift Scenarios

This repository contains code for running the post-hoc tuning method for domain drift scenarios introduced in "C. Tomani, S. Gruber, M. Erdem, D. Cremers and F. Buettner, Post-hoc Uncertainty Calibration for Domain Drift Scenarios, CVPR 2021 (Accepted - Oral)". [[Paper]](https://arxiv.org/abs/2012.10988)


## Datasets

ImageNet datasets are subject to special distributional rights. Thus, they need to be downloaded from official sources. CIFAR dataset can be downloaded with the provided code.

## Tuning & Evaluation

All post-hoc models can be tuned as well as evaluated with the `scripts/run_experiments.py` script. A path to a pre-trained classifier needs to be provided.

The evaluated metrics are stored in evaluator_storage.pkl and these results can be visualized with `scripts/plotting_experiments.ipynb`

The proposed gaussian data refiner strategy is implemented in:

`source/postevaluation/data_refiner.py` \\ \\


When calling `scripts/run_experiments.py` you can use to following arguments:

**Settings**
    The following settings are implemented as an example in `scripts/run_experiments.py`. Other settings can be implemented as well. \\
    `-settings IMAGENET_ResNet50_InD`: ResNet50 classifier for ImageNet trained on in-domain data without data refiner strategy. \\
    `-settings IMAGENET_ResNet50_OOD!!!_GN`: ResNet50 classifier for ImageNet trained on in-domain data with data refiner strategy. \\

**The following post-hoc models can be tuned:**\
    `-calib_models BaseModel`: Basic model without any calibration\
    `-calib_models TSModel`: Temperature Scaling (Guo et al., 2017)\
    `-calib_models ETSModel`: Ensemble Temperature Scaling (Zhang et al., 2020)\
    `-calib_models IRMModel`: Accuracy preserving Isotonic Regression (Zhang et al., 2020)\
    `-calib_models IROVAModel`: Isotonic Regression (Zadrozny et al., 2002)\
    `-calib_models IROVATSModel`: Isotonic Regression with Temperature Scaling (Zhang et al., 2020)\

**Additional command line arguments:**\
    `--model_path`: path to classifier model
    `--folder_path_save`: Specify where the model is stored and loaded from
    `--perturb_levels`: number of perturbation levels for tuning

    In case of ImageNet:
    `--path_data_imagenet`: path to ImageNet dataset
    `--path_data_objectnet`: path to ObjectNet dataset
    `--path_data_imagenet_corrupted`: path to corrupted ImageNet dataset

## Folder structure

The repository is structured with the following folders.

### scripts

The script `run_experiments` is used for training and evaluating the model.

### source

The python source code for all the class systems implemented in this project.

`source/data`: Data classes used in the project.\
`source/postevalutation`: All scripts for post-hoc calibration. \
`source/utils`: Utility functions that are used in the scripts.

`source/PostCalibrator.py`: Class that handels post-hoc calibration of models.\
`source/generator.py`: Preprocesses data.\
`source/model_factory.py`: Class that handels training and prediction.


## Citation:

If you find this library useful please consider citing our paper:
```
@inproceedings{tomani2021domaindriftcalibration,
    author = "Christian Tomani and Sebastian Gruber and Muhammed Ebrar Erdem and Daniel Cremers and Florian Buettner",
    title = "Post-hoc Uncertainty Calibration for Domain Drift Scenarios",
    booktitle = cvpr,
    year = "2021",
    award = "Oral Presentation"
  }
```

## References:

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning Volume 70, pages 1321–1330. JMLR. org, 2017.

Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 694–699, 2002.

Jize Zhang, Bhavya Kailkhura, and T Han. Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning. In International Conference onMachine Learning (ICML), 2020.
