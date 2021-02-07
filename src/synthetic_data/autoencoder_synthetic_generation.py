from keras import backend as K
from keras.layers import Layer


class KLDivergenceLayer(Layer):
    """
    Agregar divergencia KL a el loss del variational autoencoder
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs


def nll(y_true, y_pred):
    """ Log likelihood (Bernoulli) negativo. """
    salida = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return salida
