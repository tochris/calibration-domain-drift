import numpy as np
import tensorflow as tf


class CIFAR10:
    def __init__(
            self,
            train_batch_size=100,
            valid_batch_size=None,
            test_batch_size=None,
            shuffle_buffer=5000,
            data_path="./",
            **kwargs
    ):
        self.data_name = "CIFAR10"
        self.data_path = data_path

        self.load_data()

        self.shuffle_buffer = shuffle_buffer
        self.train_batch_size = train_batch_size
        if valid_batch_size is None:
            self.valid_batch_size = train_batch_size
        else:
            self.valid_batch_size = valid_batch_size
        if test_batch_size is None:
            self.test_batch_size = train_batch_size
        else:
            self.test_batch_size = test_batch_size

        self.prepare_generator()

    def load_data(self):
        def to_categorical(data):
            # converts to one-hot encoded vector
            data = np.array(data)
            encoded = tf.keras.utils.to_categorical(data)
            return encoded

        print("loading %s data..." % (self.data_name))
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_train, x_test = x_train / 255.0, x_test / 255.0
        y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
        y_train, y_test = to_categorical(y_train), to_categorical(y_test)
        x_valid = x_train[45000:]
        y_valid = y_train[45000:]
        x_train = x_train[:45000]
        y_train = y_train[:45000]

        self.train_data = np.reshape(x_train, (np.shape(x_train)[0], 32, 32, 3))
        self.train_label = y_train
        self.valid_data = np.reshape(x_valid, (np.shape(x_valid)[0], 32, 32, 3))
        self.valid_label = y_valid
        self.test_data = np.reshape(x_test, (np.shape(x_test)[0], 32, 32, 3))
        self.test_label = y_test

        (
            self.n_train_samples,
            self.n_rows,
            self.n_columns,
            self.n_channels_input,
        ) = self.train_data.shape
        _, self.n_classes = self.train_label.shape
        self.n_valid_samples, _, _, _ = self.valid_data.shape
        self.n_test_samples, _, _, _ = self.test_data.shape

        print(
            "n_train_samples=%d, n_valid_samples=%d, n_test_samples=%d"
            % (self.n_train_samples, self.n_valid_samples, self.n_test_samples)
        )
        print(
            "n_rows=%d, n_columns=%d, n_channels_input=%d, n_classes=%d"
            % (self.n_rows, self.n_columns, self.n_channels_input, self.n_classes)
        )
        print("*** finished loading data ***")

    def prepare_generator(self):
        img_gen_train_augment = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        self.train_dataset = tf.data.Dataset.from_generator(
            img_gen_train_augment.flow, args=[self.train_data, self.train_label, self.train_batch_size, True],
            output_types=(tf.float32, tf.float32),
            output_shapes=([self.train_batch_size, self.n_rows, self.n_columns, self.n_channels_input],
                           [self.train_batch_size, self.n_classes])
        )
        self.train_ds = self.train_dataset.take(int(self.n_train_samples / self.train_batch_size))

        img_gen_valid = tf.keras.preprocessing.image.ImageDataGenerator()
        self.valid_dataset = tf.data.Dataset.from_generator(
            img_gen_valid.flow, args=[self.valid_data, self.valid_label, self.valid_batch_size, False],
            output_types=(tf.float32, tf.float32),
            output_shapes=([self.valid_batch_size, self.n_rows, self.n_columns, self.n_channels_input],
                           [self.valid_batch_size, self.n_classes])
        )
        self.valid_ds = self.valid_dataset.take(int(self.n_valid_samples / self.valid_batch_size))

        img_gen_test = tf.keras.preprocessing.image.ImageDataGenerator()
        self.test_dataset = tf.data.Dataset.from_generator(
            img_gen_test.flow, args=[self.test_data, self.test_label, self.test_batch_size, False],
            output_types=(tf.float32, tf.float32),
            output_shapes=([self.test_batch_size, self.n_rows, self.n_columns, self.n_channels_input],
                           [self.test_batch_size, self.n_classes])
        )
        self.test_ds = self.test_dataset.take(int(self.n_test_samples / self.test_batch_size))
