import tensorflow as tf
# Run the training, save the weight and plots ?


def getOptimizer(optInfo):
    optimizer = None
    if optInfo["name"].lower()=="adam":
        optimizer = tf.keras.optimizers.Adam(
            lr=optInfo["lr"],
            beta_1=optInfo["beta_1"]
            beta_2=optInfo["beta_2"]
            epsilon=optInfo["epsilon"]
            decay=optInfo["decay"]
            amsgrad=False)

    return optimizer
