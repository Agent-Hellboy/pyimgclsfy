import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers


CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  # @param {type:"string"}

IMAGE_SHAPE = (224, 224)


class ImageNet:
    def __init__(self, image_url):
        self.grace_hopper = Image.open(image_url).resize(IMAGE_SHAPE)

        self.classifier = tf.keras.Sequential(
            [hub.KerasLayer(CLASSIFIER_URL, input_shape=IMAGE_SHAPE + (3,))]
        )

    # grace_hopper = tf.keras.utils.get_file('images.jpeg','/home')

    def get_label(self):
        self.grace_hopper = np.array(self.grace_hopper) / 255.0
        result = self.classifier.predict(self.grace_hopper[np.newaxis, ...])

        predicted_class = np.argmax(result[0], axis=-1)

        labels_path = tf.keras.utils.get_file(
            "ImageNetLabels.txt",
            "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt",
        )
        imagenet_labels = np.array(open(labels_path).read().splitlines())

        return imagenet_labels[predicted_class]
