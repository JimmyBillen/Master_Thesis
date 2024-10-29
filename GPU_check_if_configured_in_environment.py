# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# from keras.models import save_model, load_model
# from keras import utils

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

print(gpus)

print("Done")


tf.debugging.set_log_device_placement(True)

# Create some tensors
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)

# print(c)

# print("Done 2")

# MatMul: GPU:0

