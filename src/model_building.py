import tensorflow as tf
import keras_tuner

def bulid_model(hyperparameters:keras_tuner.HyperParameters):

    # define search space
    convolution_layers_total_min = 2
    convolution_layers_total_max = 10
    convolution_layers_total_step = 1

    filter_count_min = 8
    filter_count_max = 128
    filter_count_step = 8

    kernel_size_min = 2  # with value x the kernel size becomes (x, x)
    kernel_size_max = 5
    kernel_size_step = 1

    dense_layers_total_min = 1
    dense_layers_total_max = 10
    dense_layers_total_step = 1

    units_count_min = 8
    units_count_max = 128
    units_count_step = 8

    model = tf.keras.Sequential()
    
    # ------------------------
    # add convolutional layers
    # ------------------------
    # first layer
    filter_total = hyperparameters.Int("filter_total_0", 8, 128, 8)
    kernel_size = tuple([hyperparameters.Int("kernel_size",2,5,1) for _ in range(2)])
    model.add(tf.keras.layers.Conv2D(filter_total, kernel_size, activation='relu', input_shape=(150, 150, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))

    # layer 2 to n
    # TODO Variable beinhaltet nicht die erste Layer
    convolution_layers_total = hyperparameters.Int("convolution_layers_total", 2, 10, 1)
    for layer_index  in range(1, convolution_layers_total+1):
        filter_total = hyperparameters.Int(f"filter_total_{layer_index}",8, 128, 8)
        kernel_size = tuple([hyperparameters.Int(f"kernel_size_{layer_index}",2,5,1) for _ in range(2)])
        model.add(tf.keras.layers.Conv2D(filter_total, kernel_size, activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D((2,2)))

    model.add(tf.keras.layers.Flatten())

    # -------------------------------
    # add dense layers with dropout
    # -------------------------------
    dense_layers_total = hyperparameters.Int("dense_layers_total", 1, 10, 1)

    for layer_index in range(dense_layers_total):
        units = hyperparameters.Int(f"units_{layer_index}", 8, 512, 8)
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        if hp.Boolean(f"dropout_{layer_index}"):
            model.add(tf.keras.layers.Dropout(rate=0.25))

    # add final layer
    model.add(tf.keras.layers.Dense(50, activation="softmax"))

    model.compile(
        optimizer = "adam",
        loss = "categorical_crossentropy",
        metrics = "accuracy"
    )
    return model


