import keras

batch_size = 400
image_width = 224
image_height = 224
kernel_init = "he_normal"
bias_init = keras.initializers.Constant(value=0.2)
epochs = 500
num_classes = 10
