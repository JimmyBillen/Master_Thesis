import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

print(gpus)

print("Done")


tf.debugging.set_log_device_placement(True)
