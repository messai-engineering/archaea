import unittest

from numpy import ravel
from sklearn import datasets
from pybrain.datasets import ClassificationDataSet
import archaea.machine_learning.nueral_network.network_builder as builder
import archaea.machine_learning.nueral_network.nn_trainer as trainer


class NeuralNetworkIntegrationTests(unittest.TestCase):

    def neural_network_training_tests(self):
        olivetti = datasets.fetch_olivetti_faces()
        X, y = olivetti.data, olivetti.target
        print "Printing the test data X ### : " + str(X)
        print "Printing the test data Y ### : " + str(y)
        ds = ClassificationDataSet(4096, 1, nb_classes=40)
        for k in xrange(len(X)):
            ds.addSample(ravel(X[k]), y[k])
        tstdata, trndata = ds.splitWithProportion(0.25)
        trndata._convertToOneOfMany()
        tstdata._convertToOneOfMany()
        dimension = [trndata.indim, 64, trndata.outdim]
        print "Printing the dimension of NN: " + str(dimension)
        fnn = builder.NeuralNetworkBuilder(dimensions=dimension).build()
        parameters = {'network': fnn, 'dataset': trndata, 'momentum': 0.1, 'learningrate': 0.01, 'verbose': True,
                      'weightdecay': 0.01}
        bp_trainer = trainer.NeuralNetworkTrainer(parameters)
        bp_trainer.train(2)
        efficiency = bp_trainer.percentage_error_on_dataset(tstdata)
        self.assertGreaterEqual(efficiency, 90.0)

    def nn_activation_tests(self):
        olivetti = datasets.fetch_olivetti_faces()
        X, y = olivetti.data, olivetti.target
        ds = ClassificationDataSet(4096, 1, nb_classes=40)
        for k in xrange(len(X)):
            ds.addSample(ravel(X[k]), y[k])
        tstdata, trndata = ds.splitWithProportion(0.25)
        trndata._convertToOneOfMany()
        tstdata._convertToOneOfMany()
        dimension = [trndata.indim, 64, trndata.outdim]
        fnn = builder.NeuralNetworkBuilder(dimensions=dimension).build()
        parameters = {'network': fnn, 'dataset': trndata, 'momentum': 0.1,
                      'learningrate': 0.01, 'verbose': True, 'weightdecay': 0.01}
        bp_trainer = trainer.NeuralNetworkTrainer(parameters)
        bp_trainer.train(2)
        print X[0]
        value = bp_trainer.activate(X[0])
        self.assertEqual(40, len(value))
