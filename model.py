#!/usr/bin/env python

import os
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


import pdb

tf.__version__


cwd = os.getcwd()
DATA_DIR = 'images/char-1-epoch-100/train'
CHECKPOINT_DIR = os.path.join(cwd, ".model_weights")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'checkpoint')
MODEL_PATH = os.path.join(cwd, "model")
N_LABELS = 10
VALID_RATIO = 0.1
TEST_RATIO = 0.1
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128

files = os.listdir(DATA_DIR)
shape = cv2.imread(os.path.join(DATA_DIR, files[0])).shape
H, W, C = shape


def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        label, _ = filename.split("_")
        return label
    except Exception as e:
        print('error to parse %s. %s' % (filepath, e))
        return None, None


def get_data_generator(df, indices, for_training, batch_size):
    images, labels = [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, label = r['file'], r['label']
            im = cv2.imread(os.path.join(DATA_DIR, file))
            im = np.array(im) / 255.0
            images.append(np.array(im))
            labels.append(np.array([np.array(to_categorical(int(i), N_LABELS)) for i in label]))
            if len(images) >= batch_size:
                yield np.array(images), np.array(labels)
                images, labels = [], []
        if not for_training:
            break


attributes = list(map(parse_filepath, files))
CHARS_PER_IMAGE = len(attributes[0])
df = pd.DataFrame(attributes)
df['file'] = files
df.columns = ['label', 'file']

p = np.random.permutation(len(df))
test_up_to = int(len(df) * TEST_RATIO)
test_idx = p[:test_up_to]
train_idx = p[test_up_to:]

test_count = len(test_idx)
train_count = len(train_idx)
valid_count = int(train_count * VALID_RATIO)
train_count -= valid_count
train_idx, valid_idx = train_idx[:train_count], train_idx[train_count:]


print('train count: %s, valid count: %s, test count: %s' % (
    train_count, valid_count, test_count
))


def createModel():
    input_layer = tf.keras.Input(shape=(H, W, C))
    x = layers.Conv2D(32, 3, activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)

    x = layers.Dense(CHARS_PER_IMAGE * N_LABELS, activation='softmax')(x)
    x = layers.Reshape((CHARS_PER_IMAGE, N_LABELS))(x)

    model = models.Model(inputs=input_layer, outputs=x)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def showModel(model):
    model.summary()
    plot_model(model, show_shapes=True, show_layer_names=True)


def train(resume=False):
    model = createModel()
    showModel(model)
    if resume:
        model.load_weights(CHECKPOINT_PATH)
    elif not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    train_batch_size = min(TRAIN_BATCH_SIZE, train_count)
    valid_batch_size = min(VALID_BATCH_SIZE, valid_count)

    train_gen = get_data_generator(df, train_idx, for_training=True, batch_size=train_batch_size)
    valid_gen = get_data_generator(df, valid_idx, for_training=True, batch_size=valid_batch_size)

    callbacks = [
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor='accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='accuracy',
            min_delta=0.0001,
            patience=3,
            verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=train_count // train_batch_size,
        epochs=999,
        callbacks=callbacks,
        validation_data=valid_gen,
        validation_steps=valid_count // valid_batch_size,
    )

    model.save(MODEL_PATH)
    print('Training done.')

    epochs = len(history.history['loss'])
    pd.DataFrame(history.history).plot(
        figsize=(8, 5),
        grid=True,
        xticks=(np.arange(0, epochs, 1) if epochs < 10 else None),
        yticks=np.arange(0, 1, 0.1),
        ylim=(0, 1),
        xlabel='Epoch',
        ylabel='Accuracy',
    )
    plt.show()

    evaluate(model)


def evaluate(model):
    print('Evaluating...')
    test_batch_size = min(TEST_BATCH_SIZE, test_count)
    test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=test_batch_size)
    loss, accuracy = model.evaluate(test_gen, steps=test_count // test_batch_size)

    test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=test_batch_size)
    images_test, labels_test = next(test_gen)

    labels_pred = model.predict_on_batch(images_test)

    labels_true = tf.math.argmax(labels_test, axis=-1)
    labels_pred = tf.math.argmax(labels_pred, axis=-1)

    failed_idx = []
    for i, label in enumerate(labels_true):
        if labels_pred[i].numpy()[0] != label.numpy()[0]:
            failed_idx.append(i)


    print('Failed {}/{}.'.format(len(failed_idx), test_count))
    total = len(failed_idx)
    n_cols = 5
    n_rows = math.ceil(total / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    for i, img_idx in enumerate(failed_idx):
        ax = axes.flat[i]
        ax.imshow(images_test[img_idx])
        ax.set_title('pred: {}'.format(
            ''.join(map(str, labels_pred[img_idx].numpy()))))
        ax.set_xlabel('true: {}'.format(
            ''.join(map(str, labels_true[img_idx].numpy()))))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# train()
model = models.load_model(MODEL_PATH)
showModel(model)
evaluate(model)
