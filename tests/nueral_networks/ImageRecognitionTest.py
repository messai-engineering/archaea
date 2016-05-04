import os
from numpy import ravel

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

from sklearn import datasets

olivetti = datasets.fetch_olivetti_faces()
X, y = olivetti.data, olivetti.target
ds = ClassificationDataSet(4096, 1 , nb_classes=40)
for k in xrange(len(X)):
  ds.addSample(ravel(X[k]),y[k])
tstdata, trndata = ds.splitWithProportion( 0.25 )
print tstdata;
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )
print tstdata;
#fnn = buildNetwork( trndata.indim, 64 , trndata.outdim, outclass=SoftmaxLayer )
#if  os.path.isfile('oliv.xml'):
 #fnn = NetworkReader.readFrom('oliv.xml')
#else:
fnn = buildNetwork(trndata.indim, 32, 64, trndata.outdim, outclass=SoftmaxLayer )
#print fnn;
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)
NetworkWriter.writeToFile(fnn, 'oliv.xml')
trainer.trainEpochs (1)
print 'Percent Error on Test dataset: ' , percentError( trainer.testOnClassData (
           dataset=tstdata )
           , tstdata['class'] )
