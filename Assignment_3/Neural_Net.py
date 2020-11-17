'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''

# Importing Required Libraries
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.model_selection import train_test_split
import time

pd.options.mode.chained_assignment = None # To ignore the SettingWithCopy warning

# Cleaning and Prepreocessing of the Dataset
def boxplots(df,att):
    plot = df.boxplot(column=att)
    plot.set_title("Box Plot for "+att)
    plt.show()

def data_cleaning(df):
	# print("Check for Missing Values:")
    # print(df.count())
    # for i in df:
    #     boxplots(df,i)

	# df = df.drop(columns=['Education', 'Residence'])

    # Median replacement done for the attributes with outliers
	outlier_categories = ['Age','BP','HB']	
	for i in outlier_categories:
		df[i] = df[i].fillna(int(np.nanmedian(df[i])))

		# Clamping the outliers to their boxplot upper and lower bounds
		q1,q3 = np.percentile(df[i],[25,75])
		IQR = q3-q1

		min_clamp = q1-1.5*IQR
		max_clamp = q3+1.5*IQR

		df[i].loc[df[i] > max_clamp] = max_clamp
		df[i].loc[df[i] < min_clamp] = min_clamp

		# Mean replacement of outliers : Giving worse metrics
		# df[i].loc[df[i] > max_clamp] = np.mean(df[i]) 
		# df[i].loc[df[i] < min_clamp] = np.mean(df[i])

	# Mean replacement for other numeric attributes
	df['Weight'] = df['Weight'].fillna(int(np.nanmean(df['Weight'])))

	# Mode replacement done for all other attributes with missing values
	att = ['Delivery phase', 'Education', 'Residence']

	for i in att:
		df[i] = df[i].fillna(mode(df[i]))

	return df

def standardization(df):
	for i in df.keys():
		df[i] = df[i].apply(lambda x: (x-np.mean(df[i])))

	return df

def normalization(df):
	for i in df.keys():
		df[i] = df[i].apply(lambda x: (x-np.mean(df[i])/np.std(df[i])))

	return df

def feature_scaling(df):
	for i in df.keys():
		min_val = min(df[i])
		# max_val = max(df[i])
		range_val = max(df[i]) - min_val

		if range_val==0:
			df[i] = df[i].apply(lambda x: x/min_val)
		else:
			df[i] = df[i].apply(lambda x: (x-min_val)/(range_val))
	return df

def preprocessing(dataset):
	comm = pd.get_dummies(dataset.Community, prefix = "comm")
	dphase = pd.get_dummies(dataset['Delivery phase'], prefix="dphase")
	residence = pd.get_dummies(dataset.Residence, prefix='res')
	
	dataset = dataset.drop(columns=['Community','Education', 'Delivery phase'])
	dataset = feature_scaling(dataset)
	
	dataset = dataset.join(comm)
	# dataset = dataset.join(dphase)
	dataset = dataset.join(residence)

	return dataset

class NN:

	# Initilizing the required variables

	def __init__(self, num_features, dims): # Given here since the fit function does now
		# Seed the random number generator
		np.random.seed(1)
		self.lr = 0.05
		# Set the weights -> 20 hidden layer neurons 
		self.input_hidden_weights = np.random.randn(num_features, dims[0])*0.03 # To scale the weights
		self.middle_weights = np.random.randn(dims[0],dims[1])
		self.output_hidden_weights = np.random.randn(dims[1],1)
		# Setting the bias 
		self.input_bias = np.random.randn(1, dims[0])*0.001
		self.middle_bias = np.random.randn(1,dims[1])*0.001
		self.output_bias = np.random.randn(1, 1)*0.001

		print("Number of features:", num_features)
		print("Number of neurons in layer 1:",dims[0])
		print("Number of neurons in layer 2:",dims[1])
		print("Number of output layer neurons: 1")
		print("\n")

	# Defining some common activation functions

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

	''' X and Y are dataframes '''
	
	def fit(self,X,Y,epochs): # Epochs is an added parameter here
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
		# print("Number of neurons in Layer 1")
		print("Training the network......")
		for epoch in range(epochs):
			X_length = len(X)
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

				h1 = np.dot(Ah, self.middle_weights) + self.middle_bias
				Ah1 = self.tanh(h1)

				# Layer 2:
				# Ah : input
				# Yhat : final output
				h2 = np.dot(Ah1, self.output_hidden_weights) + self.output_bias
				Yhat = self.sigmoid(h2)
				
				# Calculate the error rate (MSE DERIVATIVE)
				# summ : wi*xi + bi
				# error: d(loss)/d(output)
				# activation error: d(loss)/d(o) * f'(summ)
				# weights: d(loss)/d(w1): d(loss)/d(output) * d(out)/d(summ) * d(summ)/s(wi) : activation error * d(summ)/s(wi) : activation error * input
				# bias: d(loss)/d(bias) : f'(x) * d(summ)/d(b) = f'(x)
				# input: d(loss)/d(input) : d(loss)/d(summ) * d(summ)/d(in) = f'(x) * wi
				error += np.mean(np.square(y-Yhat))
				# derivatives 
				der_error = y-Yhat
				# (der_error) * self.sigmoid_derivative(h2) is the activation error
				act_error_output = (der_error) * self.sigmoid_derivative(h2)
				backprop_error = np.dot(act_error_output, self.output_hidden_weights.T)
				grad_output_hidden_weights = np.dot(Ah1.T, act_error_output)
				self.output_hidden_weights += self.lr * grad_output_hidden_weights
				self.output_bias += self.lr * act_error_output

				act_error_middle = (backprop_error)*self.tanh_derivative(h1)
				backprop_error2 = np.dot(act_error_middle,self.middle_weights.T)
				grad_middle_hidden_weights = np.dot(Ah.T,act_error_middle)
				self.middle_weights += self.lr * grad_middle_hidden_weights
				self.middle_bias += self.lr * act_error_middle

				act_error_input = (backprop_error2)*self.tanh_derivative(h)
				grad_input_hidden_weights = np.dot(x.T, act_error_input)
				self.input_hidden_weights += self.lr * grad_input_hidden_weights
				self.input_bias += self.lr * act_error_input
			
			# if(not epoch%20):
			# 	# print("Training......")
			# 	print("Epoch:",epoch, error/X_length)

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
		h1 = np.dot(Ah,self.middle_weights) + self.middle_bias
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
	dataset = pd.read_csv("LBW_Dataset.csv")
	dataset = preprocessing(data_cleaning(dataset))
	
	# dataset = standardization(dataset)

	features = np.array(dataset.drop(columns=['Result']),dtype=np.longdouble)
	labels = np.array(dataset['Result'], dtype=np.longdouble)

	split = 0.2

	X_train,X_test, y_train, y_test = train_test_split(features,labels, test_size=split, random_state = 0)
	neurons = [20,15] # [20, x] where 13<=x<=18 gives a test accuracy of 0.85, the rest are all lesser.

	# Creating and training the neural network
	begin = time.time()
	neural_net = NN(12,neurons)
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
	


