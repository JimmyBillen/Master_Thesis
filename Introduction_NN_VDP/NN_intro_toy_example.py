# testing Keras Neural Networks
# Using Sequantial Models
# Here we create a model and don't train it. We do give it an input and study the output of the model (without it being trained), it uses the first chosen random variables

'''
In general, it's a recommended best practice to always specify the input shape of a Sequential model in advance if you know what it is.
'''


# Set Up
import tensorflow as tf
import keras
from keras import layers

# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)

# Call model on a test input
x = tf.ones((3, 3))
y = model(x)

model.summary()
print(y)

