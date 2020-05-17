import json

import os


def get_config(config_dir):
    """
    Download .json configuration file.

    :param config_dir: str, path to configuration file.
    :return config:
    """
    with open(config_dir, 'r') as file:
        config = json.load(file)
    return config


def get_images_paths(data_root_dir, nested_dir=False):
    data_dirs_paths = [os.path.join(data_root_dir, dir_name)
                       for dir_name in os.listdir(data_root_dir)]

    if nested_dir:
        dataset_images_paths = list()

        for data_dir_path in data_dirs_paths:
            images_names = os.listdir(data_dir_path)

            images_paths = [os.path.join(data_dir_path, img_name)
                            for img_name in images_names]
            for image_path in images_paths:
                dataset_images_paths.append(image_path)
    else:
        dataset_images_paths = [os.path.join(data_root_dir, image)
                                for image in os.listdir(data_root_dir)]

    return dataset_images_paths
