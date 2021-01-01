import argparse
import random

import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Randomly flip an image.
def random_flip_left_right(image):
    return tf.image.random_flip_left_right(image)


# Randomly flip an image.
def random_flip_up_down(image):
    return tf.image.random_flip_up_down(image)


# Randomly change an image contrast.
def random_contrast(image, minval=0.6, maxval=1.4):
    r = tf.random.uniform([], minval=minval, maxval=maxval)
    image = tf.image.adjust_contrast(image, contrast_factor=r)
    return tf.cast(image, tf.uint8)


# Randomly change an image brightness
def random_brightness(image, minval=0., maxval=.2):
    r = tf.random.uniform([], minval=minval, maxval=maxval)
    image = tf.image.adjust_brightness(image, delta=r)
    return tf.cast(image, tf.uint8)


# Randomly change an image saturation
def random_saturation(image, minval=0.4, maxval=2.):
    r = tf.random.uniform((), minval=minval, maxval=maxval)
    image = tf.image.adjust_saturation(image, saturation_factor=r)
    return tf.cast(image, tf.uint8)


# Randomly change an image hue.
def random_hue(image, minval=-0.04, maxval=0.08):
    r = tf.random.uniform((), minval=minval, maxval=maxval)
    image = tf.image.adjust_hue(image, delta=r)
    return tf.cast(image, tf.uint8)


def tf_rotate(input_image, min_angle = -np.pi/2, max_angle = np.pi/2):
    distorted_image = tf.expand_dims(input_image, 0)
    random_angles = tf.random.uniform(shape=(tf.shape(distorted_image)[0],), minval = min_angle , maxval = max_angle)
    distorted_image = tf.contrib.image.transform(
    distorted_image,
    tf.contrib.image.angles_to_projective_transforms(
      random_angles, tf.cast(tf.shape(distorted_image)[1], tf.float32), tf.cast(tf.shape(distorted_image)[2], tf.float32)
    ))
    rotate_image = tf.squeeze(distorted_image, [0])
    return rotate_image

# Apply all transformations to an image.
# That is a common image augmentation technique for image datasets, such as ImageNet.
def transform_image(image):
    if random.randint(1, 3) == 2:
        image = image.transpose((1, 0, 2))
    image = random_flip_left_right(image)
    image = random_flip_up_down(image)
    if random.randint(1, 21) == 11:
        image = tf_rotate(image)
    image = random_contrast(image)
    image = random_brightness(image)
    image = random_hue(image)
    image = random_saturation(image)
    return image


# Resize transformed image to a 256x256px square image, ready for training.
def resize_image(image):
    image = tf.image.resize(image, size=(args.size, args.size), preserve_aspect_ratio=False)
    image = tf.cast(image, tf.uint8)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pic_path', default='D:/pycharm_project/finalTermProject/processed_data/type1/pic2.jpg', type=str)
    parser.add_argument('-size', default=100, type=int)
    args = parser.parse_args()

    # Load image to numpy array.
    img = PIL.Image.open(args.pic_path)
    img.load()
    img_array = np.array(img)

    # Display image.
    PIL.Image.fromarray(img_array)

    # Create TensorFlow session.
    session = tf.Session()

    # Display fully pre-processed image.
    transformed_img = transform_image(img_array)
    plt.figure("fully pre-processed image")
    plt.imshow(PIL.Image.fromarray(transformed_img.eval(session=session)))

    # Display resized image.
    plt.figure("resized image")
    plt.imshow(PIL.Image.fromarray(resize_image(transformed_img).eval(session=session)))
    plt.show()