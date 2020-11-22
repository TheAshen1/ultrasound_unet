import argparse
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from datetime import datetime
from tqdm import tqdm
from data import load_data, tf_dataset, read_image_test
from model import build_model


def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return 1 - dice


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


def calculate_steps(dataset, batch_size):
    steps = len(dataset) // batch_size
    if len(dataset) % batch_size != 0:
        steps += 1
    return steps


if __name__ == '__main__':

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.RING)

    ap = argparse.ArgumentParser()

    ap.add_argument("--epochs", required=False, help="number of epochs", type=int, default=20)
    ap.add_argument("--learning_rate", required=False, help="learning rate", type=float, default=1e-4)
    ap.add_argument("--batch_size", required=False, help="batch size", type=int, default=8)
    ap.add_argument("--output_dir", required=False, help="output directory", type=str, default='results')
    args = vars(ap.parse_args())

    epochs = args['epochs']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    output_dir = args['output_dir']

    ROOT_DIR = os.path.abspath(os.curdir)
    OUTPUT_DIR = os.path.join(ROOT_DIR, output_dir, datetime.now().strftime("%d-%m-%YT%H-%M-%S"))
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(PREDICTIONS_DIR):
        os.makedirs(PREDICTIONS_DIR)

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data()

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    with strategy.scope():
        model = build_model()
        opt = tf.keras.optimizers.Adam(learning_rate)
        metrics = ['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou, dice_coefficient]
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

    callbacks = [
        ModelCheckpoint(os.path.join(OUTPUT_DIR, 'model.h5')),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        CSVLogger(os.path.join(OUTPUT_DIR, 'data.csv')),
        TensorBoard(log_dir=OUTPUT_DIR, write_images=True),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]

    train_steps = calculate_steps(train_x, batch_size)
    valid_steps = calculate_steps(valid_x, batch_size)
    test_steps = calculate_steps(test_x, batch_size)

    model.fit(train_dataset,
              validation_data=valid_dataset,
              epochs=epochs,
              steps_per_epoch=train_steps,
              validation_steps=valid_steps,
              callbacks=callbacks)

    # load model from file
    # with CustomObjectScope({'iou': iou, 'dice_coefficient': dice_coefficient}):
    #     model = tf.keras.models.load_model("model.h5")

    model.evaluate(test_dataset, steps=test_steps)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image_test(x)
        y = read_image_test(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 1)) * 255.0
        all_images = [
            x * 255.0, white_line,
            y * 255.0, white_line,
            y_pred * 255.0
        ]
        image = np.concatenate(all_images, axis=1)

        if not cv2.imwrite(os.path.join(PREDICTIONS_DIR, f'{i}.png'), image):
            raise Exception('Could not write image')

