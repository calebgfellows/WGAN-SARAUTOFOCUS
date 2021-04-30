from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf




def get_dataset_path(dataset_name,path):
    x = Path(path) / dataset_name
    return x

def load_image(image_path, size):
        load_image = Image.open(image_path)
        load_image = np.array(load_image)

        if (load_image.shape == (size, size)):

            image = np.zeros((size, size, 3))
            image[:, :, 0] = load_image[:, :]
            image[:, :, 1] = load_image[:, :]
            image[:, :, 2] = load_image[:, :]
            image = (image - 127.5) / 127.5
        else:
            image = (load_image - 127.5) / 127.5

        return image




def image_dataset(images_paths,size):
    image_array = []
    for path in images_paths:
        image = load_image(path, size)
        image_array.append(image)
    dataset = tf.data.Dataset.from_tensor_slices(image_array)


    return dataset

def load(
    dataset_name,
    mode="train",
    batch_size=100,
    shuffle=False,
    path="Random",
    size=32):
    dataset_path = get_dataset_path(dataset_name,path) / mode
    Org_images_path = [str(path) for path in dataset_path.glob("Org/*.png")]
    Cor_images_path = [str(path) for path in dataset_path.glob("Cor/*.png")]


    Org_dataset = image_dataset(Org_images_path, size)
    Cor_dataset = image_dataset(Cor_images_path, size)

    dataset = tf.data.Dataset.zip((Org_dataset, Cor_dataset))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=200)


    if (batch_size != "length"):
        dataset = dataset.batch(batch_size, drop_remainder=True)


    return dataset, len(Org_images_path)

