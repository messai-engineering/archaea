import pickle


class PyBrainNetworkPersistenceHelper:
    def __init__(self):
        pass

    @staticmethod
    def get_model_state(model_object):
        """
        Method that gets the current state of the network

        :param model_object:
        :return:
        """
        return pickle.dumps(model_object)

    @staticmethod
    def initialize_model_with_state(dom):
        """
        Method re-initiates the network from dom

        :param dom:
        :return:
        """
        return pickle.loads(dom)
