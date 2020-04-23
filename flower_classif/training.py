import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import backend as K
from sklearn.model_selection import train_test_split
import scipy.io
import math
import tarfile

import os
import keras_utils
from keras_utils import reset_tf_session

from flower_classif.image_preprocessing import prepare_raw_bytes_for_model
from flower_classif.image_preprocessing import decode_image_from_raw_bytes
from flower_classif.image_preprocessing import read_raw_from_tar

IMG_SIZE = 250


# batch generator
BATCH_SIZE = 32

# fine-tuning of an Inception V3 CNN model, previously trained on ImageNet


def get_all_filenames(tar_fn):
    with tarfile.open(tar_fn) as f:
        return [m.name for m in f.getmembers() if m.isfile()]


# list all files in tar sorted by name
all_files = sorted(get_all_filenames("102flowers.tgz"))
# read class labels (0, 1, 2, ...)
all_labels = scipy.io.loadmat('imagelabels.mat')['labels'][0] - 1
# all_files and all_labels are aligned now
N_CLASSES = len(np.unique(all_labels))


# will yield raw image bytes from tar with corresponding label
def raw_generator_with_label_from_tar(tar_fn, files, labels):
    label_by_fn = dict(zip(files, labels))
    with tarfile.open(tar_fn) as f:
        while True:
            m = f.next()
            if m is None:
                break
            if m.name in label_by_fn:
                yield f.extractfile(m).read(), label_by_fn[m.name]


def batch_generator(items, batch_size):
    """
    Batch generator that yields items in batches of size batch_size.
    There's no need to shuffle input items, just chop them into batches.
    Remember about the last batch that can be smaller than batch_size!
    Input: any iterable (list, generator, ...).
        In case of generator you can pass through your items only once!
    Output: In output yield each batch as a list of items.
    """
    count = 0
    batch = []
    for item in items:
        count += 1
        batch.append(item)
        if count >= batch_size:
            yield batch
            count = 0
            batch = []
    if count > 0:
        yield batch


def train_generator(files, labels):
    while True:  # so that Keras can loop through this as long as it wants
        for batch in batch_generator(raw_generator_with_label_from_tar(
                "102flowers.tgz", files, labels), BATCH_SIZE):
            # prepare batch images
            batch_imgs = []
            batch_targets = []
            for raw, label in batch:
                img = prepare_raw_bytes_for_model(raw)
                batch_imgs.append(img)
                batch_targets.append(label)
            # stack images into 4D tensor [batch_size, img_size, img_size, 3]
            batch_imgs = np.stack(batch_imgs, axis=0)
            # convert targets into 2D tensor [batch_size, num_classes]
            batch_targets = keras.utils.np_utils.to_categorical(batch_targets,
                                                                N_CLASSES)
            yield batch_imgs, batch_targets


def inception(use_imagenet=True):
    # load pre-trained model graph, don't add final layer
    model = keras.applications.InceptionV3(include_top=False,
                                           input_shape=(IMG_SIZE,
                                                        IMG_SIZE, 3),
                                           weights='imagenet'
                                           if use_imagenet else None)
    # add global pooling just like in InceptionV3
    new_output = keras.layers.GlobalAveragePooling2D()(model.output)
    # add new dense layer for our labels
    new_output = keras.layers.Dense(N_CLASSES,
                                    activation='softmax')(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    return model


if __name__ == '__main__':

    # test cropping
    raw_bytes = read_raw_from_tar("102flowers.tgz", "jpg/image_00001.jpg")
    img = decode_image_from_raw_bytes(raw_bytes)
    print(img.shape)
    plt.imshow(img)
    plt.show()
    img = prepare_raw_bytes_for_model(raw_bytes, normalize_for_model=False)
    print(img.shape)
    plt.imshow(img)
    plt.show()

    prepare for training

    read all filenames and labels for them

    read filenames firectly from tar
    list all files in tar sorted by name
    l_files = sorted(get_all_filenames("102flowers.tgz"))
    read class labels (0, 1, 2, ...)
    l_labels = scipy.io.loadmat('imagelabels.mat')['labels'][0] - 1
    all_files and all_labels are aligned now
    CLASSES = len(np.unique(all_labels))
    int(N_CLASSES)

    split into train/test
    (_files, te_files,
     tr_labels, te_labels) = train_test_split(all_files,
                                              all_labels,
                                              test_size=0.2,
                                              random_state=42,
                                              stratify=all_labels)

    test training generator
    r _ in train_generator(tr_files, tr_labels):
        print(_[0].shape, _[1].shape)
        plt.imshow(np.clip(_[0][0] / 2. + 0.5, 0, 1))
        break

    # remember to clear session if you start building graph from scratch!
    s = reset_tf_session()
    # don't call K.set_learning_phase() !!!
    # (otherwise will enable dropout in train/test simultaneously)

    model = inception()
    model.summary()

    # set all layers trainable by default
    for layer in model.layers:
        layer.trainable = True
        if isinstance(layer, keras.layers.BatchNormalization):
            # we do aggressive exponential smoothing of batch norm
            # parameters to faster adjust to our new dataset
            layer.momentum = 0.9

    # fix deep layers (fine-tuning only last 50)
    for layer in model.layers[:-50]:
        # fix all but batch norm layers, because we neeed
        # to update moving averages for a new dataset!
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    # compile new model
    model.compile(
        # we train 102-way classification
        loss='categorical_crossentropy',
        # we can take big lr here because we fixed first layers
        optimizer=keras.optimizers.adamax(lr=1e-2),
        metrics=['accuracy']  # report accuracy during training
