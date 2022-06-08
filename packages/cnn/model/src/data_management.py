from pathlib import Path
import pandas as pd
from model.config.core import config


def load_images(data_path) -> pd.DataFrame:
    images_df = list() # list with dataframes (path, target)

    for class_folder_path in Path.iterdir(data_path): # iter subdirectories
        if class_folder_path.is_dir(): # check if directory
            for image in Path.iterdir(class_folder_path): # iter files
                if image.is_file(): # check if file
                    if image.suffix in config.model_config.valid_image_extensions: # check if image
                        tmp = pd.DataFrame([str(image), str(class_folder_path.name)]).T
                        tmp.columns = config.model_config.data_columns
                        images_df.append(tmp)
    
    images_df = pd.concat(images_df, axis=0, ignore_index=True)
    return images_df


DATA_PATH = Path.cwd().resolve().parent / "cnn/data" / config.app_config.data_folder
tmp = load_images(DATA_PATH)


