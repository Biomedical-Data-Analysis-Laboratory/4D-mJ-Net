import constants
from callback import CollectBatchStats

import tensorflow as tf
# Run the training, save the weight and plots ?

################################################################################
# Return the optimizer based on the setting
def getOptimizer(optInfo):
    optimizer = None
    if optInfo["name"].lower()=="adam":
        optimizer = tf.keras.optimizers.Adam(
            lr=optInfo["lr"],
            beta_1=optInfo["beta_1"],
            beta_2=optInfo["beta_2"],
            epsilon=None if optInfo["epsilon"]=="None" else optInfo["epsilon"],
            decay=optInfo["decay"],
            amsgrad=False)
    # elif optInfo["name"].lower()=="sgd":
    #     ...
    #     TODO!

    return optimizer

################################################################################
# Fit the model
def fitModel(model, dataset, epochs, class_weights, sample_weights):
    batch_stats = CollectBatchStats()
    callback=[batch_stats]

    training = model.fit(dataset["train"]["data"],
                dataset["train"]["labels"],
                epochs=epochs,
                callbacks=callback,
                shuffle=True,
                validation_data=(dataset["val"]["data"], dataset["val"]["labels"]),
                sample_weight=sample_weights,
                verbose=constants.getVerbose())

    return training
