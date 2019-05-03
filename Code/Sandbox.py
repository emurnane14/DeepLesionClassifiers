##
## name: Sandbox.py
## purpose: Sandbox for experimentation
## date: 04/24/2019
##
##

from DLDataTank import DLDataTank
from Layer import Layer
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from time import clock

class Sandbox:
	def __init__(self):
		self.lesionThreshold = 0.5
	# initDLDataTank - Initialize DLDataTank
	# @param - csv file, path
	# @param - files, list of file paths with medical images
	# @param - urls, list of link locations
	# @param - outputDir, file path to directory
	# @param - length, int length of image array
	# @param - width, int width of image array
	def initDLDataTank(self, 
		csv,
		files, 
		urls, outputDir,
		length, width):
		self.csv = deepcopy(csv)
		self.files = deepcopy(files)
		self.urls = deepcopy(urls)
		self.outputDir = deepcopy(outputDir)
		self.length = deepcopy(length)
		self.width = deepcopy(width)
		self.datatank = DLDataTank(csv)
		self.datatank.retrieve(files)
		self.datatank.load(urls, outputDir) 
	# initFFBP - Initialize FFBP
	# @param - list, positive int, list of layer nodes 
	# @param - double, step for learning
	# Example: [3, 7, 7] - 3 nodes (Layer 1), 7 nodes (Layer 2), 7 nodes (Layer 3)
	def initFFBP(self,
		hiddenLayers,
		eta):
		try:
			self.hiddenLayers = deepcopy(hiddenLayers)
			self.layers = []
			## Set hidden layers
			inputLayer = Layer(False,
						   self.hiddenLayers[0],
						   np.random.rand(self.hiddenLayers[0],self.length*self.width),
						   np.random.rand(self.hiddenLayers[0]))
			self.layers.append(inputLayer)
			self.layersLen = len(self.hiddenLayers)
			for l in range(1,self.layersLen):
				layer = Layer(False,
						   self.hiddenLayers[l],
						   np.random.rand(self.hiddenLayers[l],self.hiddenLayers[l-1]),
						   np.random.rand(self.hiddenLayers[l]))
				self.layers.append(layer)
			## Set output layer
			outputLayer = Layer(True,
						   1,
						   np.random.rand(1,self.hiddenLayers[-1]),
						   np.random.rand(1))
			self.layers.append(outputLayer)
			self.layersLen += 1
			self.eta = eta
		except:
			raise
	# runExperiment - Run training-validation-testing scheme
	# @param - int, Training set length
	# @param - int, Validation set length
	# @param - int, Testing set length
	# @param - int, Maximum number of epochs
	# @param - int, Error threshold for pausing validation	
	def runExperiment(self,
		trainingLen,
		validationLen,
		testingLen,
		maxEpochs,
		errorThreshold):
		try:
			self.trainingLen = deepcopy(trainingLen)
			self.validationLen = deepcopy(validationLen)
			self.testingLen = deepcopy(testingLen)
			self.maxEpochs = deepcopy(maxEpochs)
			self.errorThreshold = deepcopy(errorThreshold)
			## Create training, validation, testing sets
			self.dataset = self.datatank.generate(self.length,
				self.width,
				self.trainingLen
				+ self.validationLen
				+ self.testingLen)
			self.training = deepcopy(self.dataset[0:self.trainingLen])
			print("Training set:")
			for tr in self.training:
				print([tr[2]])
			self.validation = deepcopy(self.dataset[self.trainingLen:self.trainingLen+self.validationLen])
			print("Validation set:")
			for v in self.validation:
				print(v[2])
			self.testing = deepcopy(self.dataset[self.trainingLen+self.validationLen:self.trainingLen+self.validationLen+self.testingLen])
			print("Testing set:")
			for te in self.testing:
				print(te[2])
			## Validation error
			validationError = 0
			## Timing value
			self.computingTime = []
			## Loop through epochs
			for epoch in range(self.maxEpochs):
				## Perform online training
				errors = []
				print("Training!")
				## Start training time 
				self.start_time = clock()
				for tr in self.training:
					## Get inputs to each layer
					inpTr = self.normalize(tr[1])
					inputsTr = [inpTr]
					for l in range(0, self.layersLen):
						inpTr = self.layers[l].get_layer_output_vector(inpTr)
						inputsTr.append(inpTr)
					## Calculate delta values
					for l in range(self.layersLen-1, 0, -1):
						if l==self.layersLen-1:
							error = self.layers[l].get_error_vector([tr[0]])
							big_e = 0.5 * error[0] ** 2
							errors.append(big_e)
							self.layers[l].set_output_layer_delta_values()
						else:
							self.layers[l].set_hidden_layer_delta_values(self.layers[l+1])
					## Calculate layer delta weights
					for l in range(self.layersLen-1, 0, -1):
						self.layers[l].calc_layer_delta_weights(self.eta)
					## Update weights
					for l in range(self.layersLen-1, 0, -1):
						self.layers[l].update_layer_weights()
				## Finish training time
				self.end_time = clock()
				self.computingTime.append(self.end_time-self.start_time)
				## Average training error
				print("Average training error: "+str(sum(errors)/self.trainingLen))
				## Perform validation
				errors = []
				print("Validation!")
				for v in self.validation:
					## Get inputs to each layer
					inpV = self.normalize(v[1])
					for l in range(0, self.layersLen):
						inpV = self.layers[l].get_layer_output_vector(inpV)
					## Calculate error
					error = self.layers[self.layersLen-1].get_error_vector([v[0]])
					big_e = 0.5 * error[0] ** 2
					errors.append(big_e)
				## Get average validation error
				epochValidationError = sum(errors)/self.validationLen
				print("Average validation error: "+str(epochValidationError))
				## Terminate training, validation 
				## if epoch is not 0 errorThreshold is greater than error or greater than previous iteration then stop
				if epoch != 0 and (epochValidationError < self.errorThreshold or epochValidationError >= validationError):
					break
				else:
					validationError = epochValidationError
			## Perform testing
			self.true = []
			self.pred = []
			self.errors = []
			print("Testing!")
			for te in self.testing:
				## Get true value
				self.true.append(te[0])
				## Get inputs to each layer
				inpTe = self.normalize(te[1])
				for l in range(0, self.layersLen):
					inpTe = self.layers[l].get_layer_output_vector(inpTe)
				## Calculate error
				error = self.layers[self.layersLen-1].get_error_vector([te[0]])
				big_e = 0.5 * error[0] ** 2
				self.errors.append(big_e)
				if inpTe[0] >= self.lesionThreshold:
					self.pred.append(1.0)
				else:
					self.pred.append(0.0)
			print("Average testing error: "+str(sum(errors)/self.testingLen))
		except:
			raise
	# getConfusionMatrix - Get confusion matrix values
	# @return - Confusion matrix
	def getConfusionMatrix(self):
		## Create confusion matrix
		cm = confusion_matrix(self.true, self.pred)
		## Plot confusion matrix
		'''
		plt.matshow(cm)
		plt.title('Confusion matrix')
		plt.colorbar()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.show()
		'''
		return cm
	# getTiming - Get computational timing
	# @return - Computing time array
	def getComputingTime(self):
		return self.computingTime
	# setLesionThreshold - Set the lesion threshold
	# @param - float, [0.0,1.0], threshold between non-lesion, lesion
	def setLesionThreshold(self,
		lesionThreshold):
		self.lesionThreshold = lesionThreshold
	# normalize - Normalization scheme for input values
	# @return - float, array data
	def normalize(self,
		inp):
		## Had to add normTerm to deal with activity function getting
		## too large
		normTerm = 256.0/2
		return [i-normTerm for i in inp]
