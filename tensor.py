import tensorflow as tf

import tensorflow_hub as hub
from tensorflow.keras import layers

classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

import numpy as np
import PIL.Image as Image
#grace_hopper = tf.keras.utils.get_file('images.jpeg','/home')

grace_hopper = Image.open('monkey.jpg').resize(IMAGE_SHAPE)



grace_hopper = np.array(grace_hopper)/255.0

print(type(grace_hopper))
result = classifier.predict(grace_hopper[np.newaxis, ...])


predicted_class = np.argmax(result[0], axis=-1)


labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

print(imagenet_labels[predicted_class])
