import keras
from sklearn.metrics import roc_auc_score
import sys
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np


class Histories(keras.callbacks.Callback):
	def __init__(self, isCenterloss):
		self.isCenterloss = isCenterloss

	def on_train_begin(self, logs={}):
		self.aucs = []
		self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		
		print('\n=========')
		print(len(self.validation_data)) #be careful of the dimenstion of the self.validation_data, somehow some extra dim will be included
		print(self.validation_data[0].shape)
		print(self.validation_data[1].shape)
		print('=========')
		#(IMPORTANT) Only use one input: "inputs=self.model.input[0]"
		ip1_input = self.model.input #this can be a list or a matrix. 
		if self.isCenterloss:
			ip1_input = self.model.input[0]
			labels = self.validation_data[1].flatten() # already are single value ground truth labels
		else:
			labels = np.argmax(self.validation_data[1],axis=1) #make one-hot vector to index for visualization
		
		ip1_layer_model = Model(inputs=ip1_input, outputs=self.model.get_layer('ip1').output)
		ip1_output = ip1_layer_model.predict(self.validation_data[0])
		
		visualize(ip1_output,labels,epoch)
		
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return


def visualize(feat, labels, epoch):

    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    XMax = np.max(feat[:,0]) 
    XMin = np.min(feat[:,1])
    YMax = np.max(feat[:,0])
    YMin = np.min(feat[:,1])

    plt.xlim(xmin=XMin,xmax=XMax)
    plt.ylim(ymin=YMin,ymax=YMax)
    plt.text(XMin,YMax,"epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)
