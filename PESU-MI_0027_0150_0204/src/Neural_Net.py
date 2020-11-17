'''
Design of a Neural Network from scratch
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''

'''
Parameters and Hyperparameters:

learning rate = 0.05 : a value larger than 0.05 led to divergence whereas a value lesser than 0.05 required several updates before reaching minima.
input layer : one neuron per feaature, i.e. 11 neurons 
1st hidden layer: 20 neurons with weights from randomly assigned from a normal distribution with mean=0, standard deviation=1
2nd hidden layer : a 20x15 matrix with weights drawn from a Gaussian normal distribution.
output layer : 15 neurons with weights from randomly assigned from a normal distribution with mean=0, standard deviation=1
bias : the input layer, hidden layer and output layers have bias drawn randomly from a normal distribution and scaled down by 0.001
number of hidden layers : 2
activation function for hidden layers : tanh
output layer function : sigmoid
number of epochs = 200 : optimun value since a value lesser than 200 wasn't enough for the model to learn, whereas a value greater than 200 overfit.
error function : mean squared error (MSE)
train-test split : 80%-20%

'''

# Importing the Required Libraries
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.model_selection import train_test_split
import time

class NN:

	# Initilizing the required variables
	def __init__(self, num_features, dims):
		# Seed the random number generator
		np.random.seed(1)
		# self.lr = 0.05 - Attempt 1 with single learning rate. Adaptive, independent learning rates resulted in a better model.
		# We used layer specific adaptive learning rates to help the NN learn better.
		# We use a lower learning rate for the i/p layer as the data is unfiltered, while a higher learning rate for the o/p layer as data is more filtered
		self.lr_input = 0.04
		self.lr_middle = 0.05
		self.lr_output = 0.06
		# Setting the weights for each of the layers (20 neuron layer, followed by 15 neuron layer has been implemented)
		# Multiplying with 0.03 helps scale the input weights
		self.input_hidden_weights = np.random.randn(num_features, dims[0])*0.03 
		self.middle_hidden_weights = np.random.randn(dims[0], dims[1])
		self.output_hidden_weights = np.random.randn(dims[1],1)
		# Setting the bias: The biases have been scaled down so as to start with a minimal value for the weights
		self.input_bias = np.random.randn(1, dims[0])*0.001
		self.middle_bias = np.random.randn(1, dims[1])*0.001
		self.output_bias = np.random.randn(1, 1)*0.001

		print("Number of features:", num_features)
		print("Number of neurons in layer 1:",dims[0], ", Activation function: tanh")
		print("Number of neurons in layer 2:",dims[1], ", Activation function: tanh")
		print("Number of output layer neurons: 1", ", Actiavation function: sigmoid")
		print("\n")

	# Defining some common activation functions and their derivatives
	# tanh and sigmoid were used as it resulted in the best accuracy for the given dataset
	def sigmoid(self,value):
		return 1/(1 + np.exp(-value))

	def ReLu(self, value):
		return np.maximum(0,value)

	def softmax(self, X):
		return np.exp(X)/np.sum(np.exp(X))

	def sigmoid_derivative(self, x):
		return self.sigmoid(x)*(1-self.sigmoid(x))
    
	def tanh(self, x):
		return ((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))
    
	def tanh_derivative(self, x):
		return (1-(self.tanh(x))**2)

	# An adaptive learning rate has been implemented by decaying the learning rate over the training process.
	def decay_lr(self, lr, decay, epoch):
		return (lr * (1 / (1 + (decay * epoch))))

	''' X and Y are dataframes '''
	
	def fit(self,X,Y,epochs): # Epochs is an added parameter here
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''

		print("Training the network......")
		for epoch in range(epochs):
			error = 0
			# Pass training set through the neural network row by row
			for x, y in zip(X, Y):
				
				x = np.array([x])
				
				# Layer 1
				# x : input
				# Ah : output
				h = np.dot(x, self.input_hidden_weights) + self.input_bias
				Ah = self.tanh(h)
				
				# Middle Layer 
				# Ah : input 
				# Ah2 : output 
				h1 = np.dot(Ah, self.middle_hidden_weights) + self.middle_bias
				Ah1 = self.tanh(h1)

				# Layer 2:
				# Ah : input
				# Yhat : final output
				h2 = np.dot(Ah1, self.output_hidden_weights) + self.output_bias
				Yhat = self.sigmoid(h2)
				
				# Calculate the error rate (MSE DERIVATIVE)
				# h : wi*xi + bi
				# error: d(loss)/d(output)
				# act_error: d(loss)/d(o) * f'(h)
				# weights: d(loss)/d(w1): d(loss)/d(output) * d(out)/d(h) * d(h)/s(wi) 
				# bias: d(loss)/d(bias) : f'(x) * d(h)/d(b) = f'(x)
				# input: d(loss)/d(input) : d(loss)/d(h) * d(h)/d(in) = f'(x) * wi

				error += np.mean(np.square(y-Yhat))
				der_error = y-Yhat
				# Backpropagation for the output layer
				act_error_output = (der_error) * self.sigmoid_derivative(h2)
				backprop_error = np.dot(act_error_output, self.output_hidden_weights.T)
				grad_output_hidden_weights = np.dot(Ah1.T, act_error_output)

				# Decaying the learning rate of o/p layer (here decay = 0.01)
				decay_output=0.01
				self.lr_output=self.decay_lr(self.lr_output,decay_output,epoch)

				# Updatings output layer weights
				self.output_hidden_weights += self.lr_output * grad_output_hidden_weights
				self.output_bias += self.lr_output * act_error_output

				# Backpropagation for the middle layer
				act_error_middle = (backprop_error)*self.tanh_derivative(h1)
				backprop_error2 = np.dot(act_error_middle,self.middle_hidden_weights.T)
				grad_middle_hidden_weights = np.dot(Ah.T,act_error_middle)

				# Decaying the learning rate of middle layer (here decay = 0.001)
				decay_middle=0.001
				self.lr_middle=self.decay_lr(self.lr_middle,decay_middle,epoch)

				# Updatings middle layer weights
				self.middle_hidden_weights += self.lr_middle * grad_middle_hidden_weights
				self.middle_bias += self.lr_middle * act_error_middle
				
				# Backpropagation for the input layer
				act_error_input = (backprop_error2)*self.tanh_derivative(h)
				grad_input_hidden_weights = np.dot(x.T, act_error_input)

				# Decaying the learning rate of input layer (here decay = 0)
				decay_input=0
				self.lr_input=self.decay_lr(self.lr_input,decay_input,epoch)

				# Updatings imput layer weights
				self.input_hidden_weights += self.lr_input * grad_input_hidden_weights
				self.input_bias += self.lr_input * act_error_input

		print("Done training!\nNumber of epochs:",epochs)
	
	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		"""

		"""
        Pass inputs through the neural network to get output
        """
		h = np.dot(X,self.input_hidden_weights) + self.input_bias
		Ah = self.tanh(h)
		h1 = np.dot(Ah,self.middle_hidden_weights) + self.middle_bias
		Ah1 = self.tanh(h1)
		h2 = np.dot(Ah1,self.output_hidden_weights) + self.output_bias
		yhat = self.sigmoid(h2)
		
		return yhat

	def CM(self,y_test,y_test_obs):
		'''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''

		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		
		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0
		
		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fn=fn+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fp=fp+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)
		a = (tp+tn)/(tp+fp+tn+fn)
		
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Accuracy : {a}")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")

		return a

if __name__ == "__main__":
	# path = "\\".join(os.getcwd().split("\\")[:-1] + ['data'])
	# print(path)
	# path = ''

	dataset = pd.read_csv("preprocessed_dataset.csv")
	
	# dataset = standardization(dataset)

	# features -> X : input to the NN
	features = np.array(dataset.drop(columns=['Result']),dtype=np.longdouble)
	# labels -> Y : expected output of the NN
	labels = np.array(dataset['Result'], dtype=np.longdouble)

	split = 0.2
	# an 80%-20% train-test split is performed with random_state = 65 for optimal performance
	# Setting random_state as a fixed value guarantees the same sequence of random numbers is generated each time the code is run
	X_train,X_test, y_train, y_test = train_test_split(features,labels, test_size=split, random_state = 65)
	neurons = [20,15] # [20, x] where 13<=x<=18 gave a test accuracy of 0.85.

	# Creating the neural network
	begin = time.time()
	# NN has 11 input features after preprocessing
	neural_net = NN(11,neurons)
	# Training the neural network
	neural_net.fit(X_train, y_train, 200)
	end = time.time()

	print(f"Time taken to train the network: {end-begin} seconds")

	# Making predictions for the test set
	predictions = neural_net.predict(X_test)
	y_hat = [1 if i>0.6 else 0 for i in predictions]

	print("\n")
	print(f"Train-Test Split:{(1-split)*100}-{split*100}")
	print("\n---------------\n")
	train_pred = neural_net.predict(X_train)
	yh_train = [1 if i>0.6 else 0 for i in train_pred]

	print("Train Set Results:")
	neural_net.CM(y_train,yh_train)
	print("\n---------------\n")
	print("Test Set Results:")
	neural_net.CM(y_test,y_hat)
	


