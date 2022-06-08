import numpy as np
import pandas as pd
import cv2
from keras.utils import np_utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from model.config import core
from model.config.core import config
import data_management as dm


def _resize_img(images_df: pd.DataFrame, n: int, image_size: int) -> np.ndarray:
    img = cv2.imread(images_df[n])
    img = cv2.resize(img, (image_size, image_size))
    return img


class TargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, encoder=LabelEncoder()) -> None:
        self.encoder = encoder

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X = np_utils.to_categorical(self.encoder.transform(X))
        return X


class CreateDataset(BaseEstimator, TransformerMixin):

    def __init__(self, image_size=50):
        self.image_size = image_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        number_observations = len(X)

        tmp = np.zeros(
            (
                number_observations, # number observations
                self.image_size, # width
                self.image_size, # length
                3 # rgb
            ),
            dtype='float32'
        )

        for n in range(0, number_observations):
            img = _resize_img(X, n, self.image_size)
            tmp[n] = img
        
        print('Dataset Images shape: {} size: {:,}'.format(tmp.shape, tmp.size))
        return tmp


if __name__ == '__main__':
    images_df = dm.load_images(core.DATA_PATH / config.data_config.data_folder_name)
    X_train, X_test, y_train, y_test = dm.get_train_test_split(images_df)
    
    encoder = TargetEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    print(y_train)

    dataCreator = CreateDataset()
    X_train = dataCreator.transform(X_train)
