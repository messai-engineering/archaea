import machine_learning.nueral_network.network_builder as network_builder
import machine_learning.common_utils.common_constants as constants


class SimpleNeuralNetwork:
    def __init__(self, network_architecture):
        self.network_architecture = network_architecture

    def network_builder(self):
        dimensions = self.network_architecture.get_ann_dimensions()
        return network_builder.NetworkBuilderFactory(dimensions, constants.SIMPLE_ANN).build_ann()

    def get_simple_neural_network(self):
        return self.network_builder()