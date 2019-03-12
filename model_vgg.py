import tensorflow as tf
from keras.applications.vgg16 import VGG16

def model_vgg(data, model_path):

    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    data = tf.multiply(data, 255.0)-mean

    vgg16 = VGG16(input_tensor=data)
    #vgg16.summary()
    return vgg16.get_layer('fc1').output, vgg16.trainable_weights
