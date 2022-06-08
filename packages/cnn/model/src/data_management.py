import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple
from model.config import core
from model.config.core import config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_images(data_path: Path) -> pd.DataFrame:
    """
    Creates dataframe with image path and target
    """ 
    images_df = list() # list with dataframes (path, target)

    for class_folder_path in Path.iterdir(data_path): # iter subdirectories
        if class_folder_path.is_dir(): # check if directory
            for image in Path.iterdir(class_folder_path): # iter files
                if image.is_file(): # check if file
                    if image.suffix in config.data_config.valid_image_extensions: # check if image
                        tmp = pd.DataFrame([str(image), str(class_folder_path.name)]).T
                        tmp.columns = config.data_config.data_columns
                        images_df.append(tmp)
    
    images_df = pd.concat(images_df, axis=0, ignore_index=True)
    return images_df


def get_train_test_split(images_df: pd.DataFrame) -> Tuple:
    """
    Performs train test split
    """ 
    X_train, X_test, y_train, y_test = train_test_split(
        images_df[config.data_config.data_columns[0]],
        images_df[config.data_config.data_columns[1]],
        test_size=config.data_config.test_size,
        random_state=config.data_config.seed
    )

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


def save_pipeline(model) -> None:
    pass

# ToDo:
def load_pipeline() -> Pipeline:
    pass


if __name__ == '__main__':
    images_df = load_images(core.DATA_PATH / config.data_config.data_folder_name)
    print(images_df.head())

    X_train, X_test, y_train, y_test = get_train_test_split(images_df)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


