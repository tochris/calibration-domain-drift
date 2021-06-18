import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
import os
import json
import math


class Objectnet:
    def __init__(
            self,
            subset='all',
            test_batch_size=100,
            image_size=224,
            padding=32,
            seed=0,
            data_path="./",
            model="ResNet50",
            **kwargs
    ):
        """
        load ObjectNet dataset (BGR, minus mean, center crop) and
        prepare it as a tf.data object.
        Args:
        subset: string, specifies the subset to be used from the whole
            objectnet dataset (9 classes (originally 313) are excluded, which
            include multiple classes from imagenet):
                'all': all 304 classes
                'only_imagenet': only 104 classes that are also included in imagenet
                'not_imagenet': 200 classes that are not included in imagenet
        test_batch_size: int
        image_size: int, image size (hight and width) after preprocessing
        padding: 32, padding when preprocessing
        seed: 0, seed for training (if zero, random seed is chosen)
        data_path: string, path to corrupted imagenet dataset
        download: bool, True if dataset is downloaded
        model: string, model name (determines preprocessing)
        """

        if subset == 'all':
            self.data_name = "Objectnet_all"
        elif subset == 'only_imagenet':
            self.data_name = "ObjectNet_only_imagenet"
        elif subset == 'not_imagenet':
            self.data_name = "ObjectNet_not_imagenet"

        self.subset = subset
        self.data_path = data_path
        self.image_size = image_size
        self.test_batch_size = test_batch_size
        self.padding = padding
        self.seed = seed

        self.model = model

        tf.random.set_seed(self.seed)

        self.load_data()
        self.prepare_generator()

    def load_data(self):
        """load objectnet data and transform it to tf.data.Dataset"""

        def get_label_list_only_imagenet(data_pah):
            """Get list of all classes for ObjectNet_only_imagenet"""

            with open(data_pah + 'mappings/imagenet_2012_label_humanreadable_mapping.json', "r") as read_file:
                label_human = json.load(read_file)
            human_label = {}
            for key, value in label_human.items():
                human_label[value] = int(key)

            with open(data_pah + '/mappings/objectnet_to_imagenet_1k.json', "r") as read_file:
                objnet_imgnethuman = json.load(read_file)

            # delete following objectclasses:
            # Alarm_clock: 'analog clock; digital clock'
            # Bicycle: 'mountain bike, all-terrain bike, off-roader; bicycle-built-for-two, tandem bicycle, tandem'
            # Chair: "barber chair; folding chair; rocking chair, rocker"
            # "Helmet": "football helmet; crash helmet"
            # "Pen": "ballpoint, ballpoint pen, ballpen, Biro; quill, quill pen; fountain pen",
            # "Skirt": "hoopskirt, crinoline; miniskirt, mini; overskirt",
            # "Still Camera": "Polaroid camera, Polaroid Land camera; reflex camera",
            # "Tie": "bow tie, bow-tie, bowtie; Windsor tie",
            # "Wheel": "car wheel; paddlewheel, paddle wheel",
            del (objnet_imgnethuman['Alarm clock'])
            del (objnet_imgnethuman['Bicycle'])
            del (objnet_imgnethuman['Chair'])
            del (objnet_imgnethuman['Helmet'])
            del (objnet_imgnethuman['Pen'])
            del (objnet_imgnethuman['Skirt'])
            del (objnet_imgnethuman['Still Camera'])
            del (objnet_imgnethuman['Tie'])
            del (objnet_imgnethuman['Wheel'])

            objnet_only_imagenet_labels = list(objnet_imgnethuman.keys())

            return objnet_only_imagenet_labels

        def get_label_list_not_imagenet(data_pah):
            """ Get list of all classes for ObjectNet_not_imagenet"""
            with open(data_pah + '/mappings/folder_to_objectnet_label.json', "r") as read_file:
                objnetfolder_objnetlabel = json.load(read_file)

            with open(data_pah + '/mappings/objectnet_to_imagenet_1k.json', "r") as read_file:
                objnet_imgnethuman = json.load(read_file)

            objnetlabels = objnetfolder_objnetlabel.values()
            objnet_only_imagenet_labels = objnet_imgnethuman.keys()

            objnet_not_imagenet_labels = []
            for label in objnetlabels:
                if label not in objnet_only_imagenet_labels:
                    objnet_not_imagenet_labels.append(label)

            return objnet_not_imagenet_labels

        def get_label_list_all_imagenet(data_path):
            """Get list of all classes for ObjectNet_all_imagenet"""
            return get_label_list_only_imagenet(data_path) + get_label_list_not_imagenet(data_path)

        def get_str_label_dir_list(data_pah, label_list):
            # create list of paths to images based on class labels
            with open(data_pah + '/mappings/folder_to_objectnet_label.json', "r") as read_file:
                objnetfolder_objnetlabel = json.load(read_file)
            objnetlabel_objnetfolder = {}
            for key, value in objnetfolder_objnetlabel.items():
                objnetlabel_objnetfolder[value] = key

            str_label_list = []
            for label in label_list:
                str_label = str(data_pah + 'images/' + str(objnetlabel_objnetfolder[label]) + '/*')
                str_label_list.append(str_label)
            return str_label_list

        def get_mapping_objnetlabel_imglabel(data_path):
            """mapping directory from objectnet label to object label (int)"""
            with open(data_path + '/mappings/imagenet_2012_label_humanreadable_mapping.json', "r") as read_file:
                label_human = json.load(read_file)
            human_label = {}
            for key, value in label_human.items():
                human_label[value] = int(key)

            with open(data_path + '/mappings/objectnet_to_imagenet_1k.json', "r") as read_file:
                objnet_imgnethuman = json.load(read_file)

            objnet_only_imagenet_labels = get_label_list_only_imagenet(data_path)
            objnet_imglabel = {}
            for label_objnet in objnet_only_imagenet_labels:
                objnet_imglabel[label_objnet] = human_label[objnet_imgnethuman[label_objnet]]
            return objnet_imglabel

        def get_mapping_objnetfolder_objnetlabel(data_path):
            """mapping directory from folder name to objectnet label (string)"""
            with open(data_path + '/mappings/folder_to_objectnet_label.json', "r") as read_file:
                objnetfolder_objnetlabel = json.load(read_file)
            return objnetfolder_objnetlabel

        def get_tf_mapping_objnetfolder_imglabel(data_path):
            """mapping directory from folder name to imagenet label (string)"""
            mapping_objnetfolder_objnetlabel = get_mapping_objnetfolder_objnetlabel(data_path)
            mapping_objnetlabel_imglabel = get_mapping_objnetlabel_imglabel(data_path)
            mapping_objnetfolder_imglabel = {}
            for folder in mapping_objnetfolder_objnetlabel.keys():
                if mapping_objnetfolder_objnetlabel[folder] in list(mapping_objnetlabel_imglabel.keys()):
                    mapping_objnetfolder_imglabel[folder] = mapping_objnetlabel_imglabel[
                        mapping_objnetfolder_objnetlabel[folder]]
            table_objnetfolder_imglabel = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=tf.constant(list(mapping_objnetfolder_imglabel.keys())),
                    values=tf.constant(list(mapping_objnetfolder_imglabel.values())),
                ),
                default_value=tf.constant(np.random.randint(1000)),
                name="objnetfolder_objnetlabel"
            )
            return table_objnetfolder_imglabel

        def get_label(file_path):
            """convert the path to a list of path components"""
            parts = tf.strings.split(file_path, os.path.sep)
            # The second to last is the class-directory
            objnetlabel = self.table_objnetfolder_imglabel.lookup(parts[-2])
            return objnetlabel

        def decode_img(img):
            """convert the compressed string to a 3D uint8 tensor"""
            img = tf.image.decode_jpeg(img, channels=3)
            # Use `convert_image_dtype` to convert to floats in the [0,1] range.
            img = tf.image.convert_image_dtype(img, tf.float32)
            return img

        def process_path(file_path):
            label = get_label(file_path)
            # load the raw data from the file as a string
            img = tf.io.read_file(file_path)
            img = decode_img(img)
            return {"image": img, "label": label}

        print("loading %s data..." % (self.subset))
        if self.subset in ['all', 'only_imagenet', 'not_imagenet']:
            pass
        else:
            raise ValueError('Error: subset must be one of the following all, only_imagenet, not_imagenet')

        if self.subset == 'all':
            label_list = get_label_list_all_imagenet(self.data_path)
            str_label_list = get_str_label_dir_list(self.data_path, label_list)
        elif self.subset == 'only_imagenet':
            label_list = get_label_list_only_imagenet(self.data_path)
            str_label_list = get_str_label_dir_list(self.data_path, label_list)
        elif self.subset == 'not_imagenet':
            label_list = get_label_list_not_imagenet(self.data_path)
            str_label_list = get_str_label_dir_list(self.data_path, label_list)
        print('Number of Labels: ', len(label_list))

        self.table_objnetfolder_imglabel = get_tf_mapping_objnetfolder_imglabel(self.data_path)
        list_ds = tf.data.Dataset.list_files(str_label_list)
        self.test_dataset = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # fake numbers for consistency with other data set classes
        self.nb_train_sampels = 1281167
        self.nb_valid_sampels = 50000
        self.nb_test_sampels = 50000
        self.train_steps_per_epoch = math.floor(self.nb_train_sampels / float(self.test_batch_size))
        self.valid_steps_per_epoch = math.floor(self.nb_valid_sampels / float(self.test_batch_size))
        self.test_steps_per_epoch = math.floor(self.nb_test_sampels / float(self.test_batch_size))


        def distorted_bounding_box_crop(image,
                                        bbox,
                                        min_object_covered=0.1,
                                        aspect_ratio_range=(0.75, 1.33),
                                        area_range=(0.05, 1.0),
                                        max_attempts=100,
                                        scope=None):
            """
            Adpted from:
            https://github.com/google-research/google-research/blob/
            master/uq_benchmark_2019/imagenet/resnet_preprocessing.py (11-22-2020)

            Generates cropped_image using one of the bboxes randomly distorted.
            See `tf.image.sample_distorted_bounding_box` for more documentation.
            Args:
            image: `Tensor` of binary image data.
            bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]` where
              each coordinate is [0, 1) and the coordinates are arranged as `[ymin,
              xmin, ymax, xmax]`. If num_boxes is 0 then use the whole image.
            min_object_covered: An optional `float`. Defaults to `0.1`. The cropped area
              of the image must contain at least this fraction of any bounding box
              supplied.
            aspect_ratio_range: An optional list of `float`s. The cropped area of the
              image must have an aspect ratio = width / height within this range.
            area_range: An optional list of `float`s. The cropped area of the image must
              contain a fraction of the supplied image within in this range.
            max_attempts: An optional `int`. Number of attempts at generating a cropped
              region of the image of the specified constraints. After `max_attempts`
              failures, return the entire image.
            scope: Optional `str` for name scope.
            Returns:
            cropped image `Tensor`
            """
            shape = array_ops.shape(image)
            sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                shape,
                bounding_boxes=bbox,
                seed=self.seed,
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, _ = sample_distorted_bounding_box

            # Crop the image to the specified bounding box.
            offset_y, offset_x, _ = tf.unstack(bbox_begin)
            target_height, target_width, _ = tf.unstack(bbox_size)
            crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
            image = tf.image.crop_to_bounding_box(image,
                                                  offset_y,
                                                  offset_x,
                                                  target_height,
                                                  target_width)
            return image

        def _at_least_x_are_equal(a, b, x):
            """
            Adpted from:
            https://github.com/google-research/google-research/blob/
            master/uq_benchmark_2019/imagenet/resnet_preprocessing.py (11-22-2020)

            At least `x` of `a` and `b` `Tensors` are equal.
            """
            match = tf.equal(a, b)
            match = tf.cast(match, tf.int32)
            return tf.greater_equal(tf.reduce_sum(match), x)

        def center_crop_and_resize(image):
            """
            Adpted from:
            https://github.com/google-research/google-research/blob/
            master/uq_benchmark_2019/imagenet/resnet_preprocessing.py (11-22-2020)

            Make a center crop of image_size from image
            """
            shape = array_ops.shape(image)
            image_height = shape[0]
            image_width = shape[1]

            padded_center_crop_size = tf.cast(
                ((self.image_size / (self.image_size + self.padding)) *
                 tf.cast(tf.minimum(image_height, image_width), tf.float32)),
                tf.int32)

            offset_height = ((image_height - padded_center_crop_size) + 1) // 2
            offset_width = ((image_width - padded_center_crop_size) + 1) // 2
            crop_window = tf.stack([offset_height, offset_width,
                                    padded_center_crop_size, padded_center_crop_size])

            image = tf.image.crop_to_bounding_box(image,
                                                  offset_height,
                                                  offset_width,
                                                  padded_center_crop_size,
                                                  padded_center_crop_size)
            image = tf.image.resize(image, (self.image_size, self.image_size))  # Resize the image

            return image

        def random_crop_and_resize(image):
            """
            Adpted from:
            https://github.com/google-research/google-research/blob/
            master/uq_benchmark_2019/imagenet/resnet_preprocessing.py (11-22-2020)

            Make a random crop of image_size from image
            """
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
            image = distorted_bounding_box_crop(
                image,
                bbox,
                min_object_covered=0.1,
                aspect_ratio_range=(3. / 4, 4. / 3.),
                area_range=(0.08, 1.0),
                max_attempts=10,
                scope=None)
            original_shape = array_ops.shape(image)
            bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

            image = tf.cond(
                bad,
                lambda: center_crop_and_resize(image),
                lambda: tf.image.resize(image, (self.image_size, self.image_size)))

            return image

        def preprocess_for_eval(image):
            """
            Adpted from:
            https://github.com/google-research/google-research/blob/
            master/uq_benchmark_2019/imagenet/resnet_preprocessing.py (11-22-2020)

            Preprocesses the given image for evaluation
            Args:
                image: 'Tensor', representing an image
            Returns:
                The preprocessed image tensor for evaluation
            """

            image = tf.cast(image, tf.float32)
            image = center_crop_and_resize(image)
            image = tf.reshape(image, [self.image_size, self.image_size, 3])
            return image

        def preprocess_for_train(image):
            """
            Adpted from:
            https://github.com/google-research/google-research/blob/
            master/uq_benchmark_2019/imagenet/resnet_preprocessing.py (11-22-2020)

            Preprocesses the given image for training
            Args:
                image: 'Tensor', representing an image
            Returns:
                The preprocessed image tensor for training
            """
            image = tf.cast(image, tf.float32)
            image = random_crop_and_resize(image)
            image = tf.image.random_flip_left_right(image)
            image = tf.reshape(image, [self.image_size, self.image_size, 3])
            return image

        def preprocess_image(image, is_training, use_bfloat16=False):
            """
            Preprocess an image tensor for training or testing
            Args:
                image: 'Tensor', representing an image
                is_training: 'bool', if True preprocessing for training
                use_bfloat16: 'bool', for whether to use bfloat16.
            Returns:
                The preprocessed image tensor
            """
            image = tf.cast(image, tf.float32)
            # image = tf.clip_by_value(image, 0., 255.) / 255.

            if is_training:
                image = preprocess_for_train(image)
            else:
                image = preprocess_for_eval(image)

            image = image * 255.0

            if self.model in ["ResNet50", "ResNet101", "ResNet152", "VGG19"]:
                # rescale
                mean = np.ones((self.image_size, self.image_size, 3), dtype=np.float32)
                mean[..., 0] = 123.68  # 103.939
                mean[..., 1] = 116.779  # 116.779
                mean[..., 2] = 103.939  # 123.68
                image = image - mean
                # Change from RGB to BGR
                image = image[..., ::-1]
            elif self.model in ["DenseNet121", "DenseNet169", "DenseNet201"]:
                image = tf.keras.applications.densenet.preprocess_input(image)
            elif self.model in ["EfficientNetB7", "EfficientNetB0"]:
                image = tf.keras.applications.efficientnet.preprocess_input(image)
            elif self.model in ["Xception"]:
                image = tf.keras.applications.xception.preprocess_input(image)
            elif self.model in ["MobileNetV2"]:
                image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            elif self.model in ["ResNet50_basic"]:  # for ResNet from snoek paper
                image = image / 255.0
            else:
                ValueError("No Model specified!")

            # if use_bfloat16==True convert to bfloat16
            image = tf.image.convert_image_dtype(
                image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
            return image

        self.test_dataset = self.test_dataset.map(
            lambda d: {"image": preprocess_image(d["image"], is_training=False), "label": d["label"]})

        # one_hot encoded labels
        self.test_dataset = self.test_dataset.map(
            lambda d: {"image": d["image"], "label": tf.one_hot(d["label"], 1000)})

        # transform to tuple
        self.test_dataset = self.test_dataset.map(lambda d: (d["image"], d["label"]))

    def prepare_generator(self):
        self.test_ds = self.test_dataset.batch(self.test_batch_size)
        self.test_ds.prefetch(tf.data.experimental.AUTOTUNE)
