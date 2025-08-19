import pandas as pd
from os import listdir
from PIL import Image as PImage
import numpy as np
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True


def interpolate_time(df, new_index):

    """Return a new DataFrame with all columns values interpolated to the new_index values."""
    # Convert df.index to datetime and then to numerical values (timestamps)
    df.index = pd.to_datetime(df.index)  # Convert index to datetime
    df_index_timestamp = df.index.astype(int) / 10**9  # Convert to seconds since epoch
    
    # Convert new_index to datetime if it contains date strings
    new_index_datetime = pd.to_datetime(new_index)
    new_index_timestamp = new_index_datetime.astype(int) / 10**9  # Convert new_index to timestamps

    # Create an empty DataFrame for output
    df_out = pd.DataFrame(index=new_index_datetime)
    df_out.index.name = df.index.name

    # Interpolate each column
    for colname, col in df.items():
        df_out[colname] = np.interp(new_index_timestamp, df_index_timestamp, col)

    # Convert the index of the interpolated DataFrame back to datetime
    df_out.index = new_index_datetime

    return df_out


def extract_image_paths_zurich(path_to_folders, start_folder, end_folder):
    """
    Extract paths from all the images from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.
        start_folder: start date from which images need to be extracted
        end_folder: end date until which images need to be extracted

    Returns:
        all_images (list): A list of all extracted images.
    """
    image_folders = sorted(listdir(path_to_folders)) 

    all_paths = []

    # Select the images from start until end date
    selected_folders = [folder for folder in image_folders if start_folder <= folder <= end_folder]
    selected_folders = sorted(selected_folders)

    # Save all paths from the selected folders
    for folder in selected_folders:
        subfolders=sorted(listdir(f"{path_to_folders}/{folder}"))
        for subfolder in subfolders:
            path_to_image = f"{path_to_folders}/{folder}/{subfolder}"
            images_list = listdir(path_to_image)
            for image in images_list:
                all_paths.append(f"{path_to_image}/{image}")
    return all_paths


def extract_image_paths_pileaute(path_to_folders, start_folder, end_folder, magnification=10):
    """
    Extract paths from all the images from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.
        start_folder: start date from which images need to be extracted
        end_folder: end date until which images need to be extracted
        magnification: type of magnification (10 or 40)

    Returns:
        all_images (list): A list of all extracted images.
    """
    image_folders = sorted(listdir(path_to_folders)) 

    all_paths = []

    # Select the images from start until end date
    selected_folders = [folder for folder in image_folders if start_folder <= folder <= end_folder]
    selected_folders = sorted(selected_folders)

    # Save all paths from the selected folders
    for folder in selected_folders:
        path_to_image = f"{path_to_folders}/{folder}/basin5/{magnification}x"
        images_list = sorted(listdir(path_to_image))
        for image in images_list:
            all_paths.append(f"{path_to_image}/{image}")
    return all_paths
