import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

def load_data():
    all_data = pd.read_csv("kaggle/train_masks.csv")
    all_usable_data = all_data[all_data.pixels.notnull()]

    train_dataset_info, val_dataset_info = train_test_split(all_usable_data, test_size=0.2, random_state=SEED)
    val_dataset_info, test_dataset_info = train_test_split(val_dataset_info, test_size=0.5, random_state=SEED)

    print(len(train_dataset_info))
    print(len(val_dataset_info))
    print(len(test_dataset_info))

    training = get_image_names(train_dataset_info)
    validation = get_image_names(val_dataset_info)
    test = get_image_names(test_dataset_info)

    return training, validation, test



def get_image_names(dataset_info: list):
    x = []
    y = []
    for index, row in dataset_info.iterrows():
        x.append(f'kaggle/train/{row["subject"]}_{row["img"]}.tif')
        y.append(f'kaggle/train/{row["subject"]}_{row["img"]}_mask.tif')

    return x, y

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    return x

def read_image_test(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_image(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 1])
    y.set_shape([256, 256, 1])
    return x, y


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset
