import tensorflow as tf
from tf.keras.applications.vgg16 import VGG16

def model_vgg(data, model_path):

    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    data = tf.mul(data, 255.0)-mean

    vgg16 = VGG16(input_tensor=data)

    return vgg16.outputs, vgg16.trainable_weights
