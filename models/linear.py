import tensorflow as tf


class Linear():

    def __init__(self):
        self.linear = None

    def build_model(self, setting_configuration):
        self.linear = tf.keras.Sequential(
            tf.keras.layers.Dense(units=1)
        )
        self.linear.compile(loss=setting_configuration['loss'], metrics=setting_configuration['metrics'],
                            optimizer=setting_configuration['optimizer'])

    def fit_model(self, Xtrain, Xval, epochs=20, callbacks=None):
        history = self.linear.fit(Xtrain, epochs=epochs, validation_data=Xval)
        return history

    def get_instance(self):
        return self.linear
