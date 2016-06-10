import cnn_kwargs_builder as kwargs_builder
from nolearn.lasagne import NeuralNet


class cnn_network_builder:
    def __init__(self, network_architecture):
        self.network_architecture = network_architecture

    def build(self):
        """
        Method parses the network architecture and then builds the network which can be trained using the network trainer

        :return:
        """
        [layer_tuples, kwargs] = kwargs_builder.ConvNetArgsBuilder(self.network_architecture).cnn_parameters_parser()
        cnn_network = NeuralNet(layer_tuples, **kwargs)
        return cnn_network
